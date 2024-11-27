from itertools import product
from typing import Optional

from sklearn.datasets import make_blobs

from recursive_clustering.experiment.open_ml_clustering_experiment import ClusteringExperiment


class BlobClusteringExperiment(ClusteringExperiment):
    def __init__(
            self,
            n_samples: Optional[int] = 100,
            n_features: Optional[int] = 2,
            centers: Optional[int] = 3,
            cluster_std: Optional[float] = 1.0,
            center_box: Optional[tuple[float, float]] = (-10.0, 10.0),
            shuffle: Optional[bool] = True,
            seeds_dataset: Optional[int] = 0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.cluster_std = cluster_std
        self.center_box = center_box
        self.shuffle = shuffle
        self.seeds_dataset = seeds_dataset

    def _add_arguments_to_parser(self):
        super()._add_arguments_to_parser()
        self.parser.add_argument('--n_samples', type=int, default=self.n_samples)
        self.parser.add_argument('--n_features', type=int, default=self.n_features)
        self.parser.add_argument('--centers', type=int, default=self.centers)
        self.parser.add_argument('--cluster_std', type=float, default=self.cluster_std)
        self.parser.add_argument('--center_box', type=tuple, default=self.center_box)
        self.parser.add_argument('--shuffle', type=bool, default=self.shuffle)
        self.parser.add_argument('--seeds_dataset', type=int, default=self.seeds_dataset, nargs='*')

    def _unpack_parser(self):
        args = super()._unpack_parser()
        self.n_samples = args.n_samples
        self.n_features = args.n_features
        self.centers = args.centers
        self.cluster_std = args.cluster_std
        self.center_box = args.center_box
        self.shuffle = args.shuffle
        self.seeds_dataset = args.seeds_dataset

    def _get_combinations(self):
        combinations = list(product(self.models_nickname, self.seeds_models, self.seeds_dataset))
        combination_names = ['model_nickname', 'seed_model', 'seed_dataset']
        combinations = [list(combination) + [self.models_params[combination[0]]] + [self.fits_params[combination[0]]]
                        for combination in combinations]
        combination_names += ['model_params', 'fit_params']
        unique_params = dict(n_samples=self.n_samples, n_features=self.n_features, centers=self.centers,
                             cluster_std=self.cluster_std, center_box=self.center_box, shuffle=self.shuffle)
        extra_params = dict(n_jobs=self.n_jobs, return_results=False)
        return combinations, combination_names, unique_params, extra_params

    def _load_data(self, combination: dict, unique_params: Optional[dict] = None, extra_params: Optional[dict] = None,
                   **kwargs):
        n_samples = unique_params['n_samples']
        n_features = unique_params['n_features']
        centers = unique_params['centers']
        cluster_std = unique_params['cluster_std']
        center_box = unique_params['center_box']
        shuffle = unique_params['shuffle']
        seed_dataset = combination['seed_dataset']
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                          center_box=center_box, shuffle=shuffle, random_state=seed_dataset)
        return {'X': X, 'y': y}
