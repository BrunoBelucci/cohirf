import numpy as np
import scipy.sparse as sp
from sklearn.base import _fit_context
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import _check_sample_weight, _is_arraylike_not_scalar, check_array
from sklearn.utils.extmath import row_norms
from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit, _mini_batch_step, _kmeans_plusplus
from sklearn.utils.parallel import _get_threadpool_controller
import dask.array as da
from pathlib import Path
import h5py
from dask.array.slicing import shuffle_slice
from dask.array import shuffle
from shutil import rmtree
import os
import zarr as zr


class LazyMiniBatchKMeans(MiniBatchKMeans):
    def __init__(
            self,
            n_clusters=8,
            *,
            init="k-means++",
            max_iter=100,
            batch_size=1024,
            verbose=0,
            compute_labels=True,
            random_state=None,
            tol=0.0,
            max_no_improvement=10,
            init_size=None,
            n_init="auto",
            reassignment_ratio=0.01,
            shuffle_every_n_epochs=50,
            tmp_dir=Path.cwd(),
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            tol=tol,
            n_init=n_init,
            max_no_improvement=max_no_improvement,
            batch_size=batch_size,
            compute_labels=compute_labels,
            init_size=init_size,
            reassignment_ratio=reassignment_ratio,
        )
        self.shuffle_every_n_epochs = shuffle_every_n_epochs
        self.tmp_dir = tmp_dir

    # Basically we:
    # 1 - Accept dask arrays as input by not casting them to numpy arrays until necessary
    # 2 - Transform dask arrays in numpy arrays when sampling them in init_centroids and in the minibatch step
    # 3 - Assign the final clusters per batch to the labels_ attribute
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.

            .. versionadded:: 0.20

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            cast_to_ndarray=False,
        )

        # we will ensure that X is chunked by batch_size row-wise
        if isinstance(X, da.Array):
            X = X.rechunk((self.batch_size, -1))

        self._check_params_vs_input(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()
        n_samples, n_features = X.shape

        # Validate init array
        init = self.init
        if _is_arraylike_not_scalar(init):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        self._check_mkl_vcomp(X, self._batch_size)

        # precompute squared norms of data points
        # x_squared_norms = row_norms(X, squared=True)

        # Validation set for the init
        validation_indices = random_state.randint(0, n_samples, self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]

        # perform several inits with random subsets
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            cluster_centers = self._init_centroids(
                X,
                # x_squared_norms=x_squared_norms,
                init=init,
                random_state=random_state,
                init_size=self._init_size,
                sample_weight=sample_weight,
            )

            if isinstance(X_valid, da.Array):
                X_valid_computed = X_valid.compute()
            else:
                X_valid_computed = X_valid

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid_computed,
                sample_weight_valid,
                cluster_centers,
                n_threads=self._n_threads,
            )

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: {inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        # Initialize number of samples seen since last reassignment
        self._n_since_last_reassign = 0

        n_steps = (self.max_iter * n_samples) // self._batch_size
        if isinstance(X, da.Array):
            # we do this to persist X_temp and to be able to shuffle it
            rmtree(self.tmp_dir / "X_temp.zarr", ignore_errors=True)
            X.to_zarr(self.tmp_dir / "X_temp.zarr")
            X_temp = da.from_zarr(self.tmp_dir / "X_temp.zarr")
            sample_weight_shuffled = sample_weight

        shuffle_every_n_iterations = self.shuffle_every_n_epochs * (n_samples // self._batch_size)

        with _get_threadpool_controller().limit(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_steps):
                # Sample a minibatch from the full dataset
                if isinstance(X, da.Array):
                    if i % shuffle_every_n_iterations == 0 and i != 0:
                        # iterating by block is MUCH faster than indexing, so we use the blocks and periodically
                        # shuffle them to avoid bias
                        # shuffle indexes
                        shuffled_index = random_state.permutation(n_samples)
                        sample_weight_shuffled = sample_weight_shuffled[shuffled_index]
                        X_temp = shuffle_slice(X_temp, shuffled_index)
                        zar_file = zr.open(self.tmp_dir / "X_temp.zarr", mode="a")
                        X_temp = X_temp.store(zar_file)
                        X_temp = da.from_zarr(self.tmp_dir / "X_temp.zarr")
                    block_index = random_state.randint(0, X_temp.numblocks[0])
                    start_index = block_index * self._batch_size
                    stop_index = min((block_index + 1) * self._batch_size, n_samples)
                    minibatch_indices = list(range(start_index, stop_index))
                    sample_weight_minibatch = sample_weight_shuffled[minibatch_indices]
                    X_minibatch = X_temp.blocks[block_index].compute()
                else:
                    minibatch_indices = random_state.randint(0, n_samples, self._batch_size)
                    X_minibatch = X[minibatch_indices]
                    sample_weight_minibatch = sample_weight[minibatch_indices]

                # Perform the actual update step on the minibatch data
                batch_inertia = _mini_batch_step(
                    X=X_minibatch,
                    sample_weight=sample_weight_minibatch,
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads,
                )

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers) ** 2)
                else:
                    centers_squared_diff = 0

                centers, centers_new = centers_new, centers

                # Monitor convergence and do early stopping if necessary
                if self._mini_batch_convergence(
                        i, n_steps, n_samples, centers_squared_diff, batch_inertia
                ):
                    break

            if isinstance(X, da.Array):
                rmtree(self.tmp_dir / "X_temp.zarr", ignore_errors=True)

        self.cluster_centers_ = centers
        self._n_features_out = self.cluster_centers_.shape[0]

        self.n_steps_ = i + 1
        self.n_iter_ = int(np.ceil(((i + 1) * self._batch_size) / n_samples))

        if self.compute_labels:
            # we need to iterate per batch to avoid memory errors
            if isinstance(X, da.Array):
                self.labels_ = []
                self.inertia_ = 0
                for row in range(0, X.shape[0], self._batch_size):
                    X_chunk = X[row:row + self._batch_size]
                    X_chunk = X_chunk.compute()
                    labels_chunk, inertia_chunk = _labels_inertia_threadpool_limit(
                        X_chunk,
                        sample_weight[row:row + self._batch_size],
                        self.cluster_centers_,
                        n_threads=self._n_threads,
                    )
                    self.labels_.append(labels_chunk)
                    self.inertia_ += inertia_chunk
                self.labels_ = np.concatenate(self.labels_)
            else:
                self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                    X,
                    sample_weight,
                    self.cluster_centers_,
                    n_threads=self._n_threads,
                )
        else:
            self.inertia_ = self._ewa_inertia * n_samples

        return self

    def _init_centroids(
            self,
            X,
            # x_squared_norms,
            init,
            random_state,
            sample_weight,
            init_size=None,
            n_centroids=None,
    ):
        """Compute the initial centroids.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        sample_weight : ndarray of shape (n_samples,)
            The weights for each observation in X. `sample_weight` is not used
            during initialization if `init` is a callable or a user provided
            array.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        n_centroids : int, default=None
            Number of centroids to initialize.
            If left to 'None' the number of centroids will be equal to
            number of clusters to form (self.n_clusters).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
            Initial centroids of clusters.
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters if n_centroids is None else n_centroids

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            if isinstance(X, da.Array):
                X = X.compute()
            n_samples = X.shape[0]
            sample_weight = sample_weight[init_indices]

        x_squared_norms = row_norms(X, squared=True)

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.choice(
                n_samples,
                size=n_clusters,
                replace=False,
                p=sample_weight / sample_weight.sum(),
            )
            centers = X[seeds]
        elif _is_arraylike_not_scalar(self.init):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers