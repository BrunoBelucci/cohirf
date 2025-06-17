import argparse
import os
import numpy as np
from itertools import product
from typing import Optional
import matplotlib.pyplot as plt
from cohirf.experiment.clustering_experiment import ClusteringExperiment
from cohirf.experiment.tested_models import models_dict as default_models_dict
from cohirf.models.cohirf import CoHiRF, BaseCoHiRF
from cohirf.models.batch_cohirf import BatchCoHiRF
import optuna
from sklearn.cluster import DBSCAN


models_dict = default_models_dict.copy()
models_dict.update(
	{
		CoHiRF.__name__: (
			CoHiRF,
			dict(n_features=4),
			dict(
				repetitions=optuna.distributions.IntDistribution(2, 10),
				kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
			),
			[dict(
				repetitions=5,
				kmeans_n_clusters=3,
			)],
		),
		"CoHiRF-DBSCAN": (
			BaseCoHiRF,
			dict(base_model=DBSCAN, n_features=4, repetitions=1),
			dict(
				base_model_kwargs=dict(
					eps=optuna.distributions.FloatDistribution(1e-1, 10),
					min_samples=optuna.distributions.IntDistribution(2, 50),
				),
			),
			[dict(
				base_model_kwargs=dict(
					eps=0.5,
					min_samples=5,
				),
			)],
		),
		"BatchCoHiRF": (
			BatchCoHiRF,
			dict(cohirf_kwargs=dict(n_features=4)),
			dict(
				cohirf_kwargs=dict(
					repetitions=optuna.distributions.IntDistribution(2, 10),
					kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
				)
			),
			[dict(
				cohirf_kwargs=dict(
					repetitions=5,
					kmeans_n_clusters=3,
				)
			)],
		),
		"BatchCoHiRF-1iter": (
			BatchCoHiRF,
			dict(cohirf_kwargs=dict(max_iter=1, n_features=4)),
			dict(
				cohirf_kwargs=dict(
					repetitions=optuna.distributions.IntDistribution(2, 10),
					kmeans_n_clusters=optuna.distributions.IntDistribution(2, 5),
				)
			),
			[dict(
				cohirf_kwargs=dict(
					repetitions=5,
					kmeans_n_clusters=3,
				)
			)],
		),
		"BatchCoHiRF-DBSCAN": (
			BatchCoHiRF,
			dict(
				cohirf_model=BaseCoHiRF,
				cohirf_kwargs=dict(base_model=DBSCAN, n_features=4, repetitions=1),
			),
			dict(
				cohirf_kwargs=dict(
					base_model_kwargs=dict(
						eps=optuna.distributions.FloatDistribution(1e-1, 10),
						min_samples=optuna.distributions.IntDistribution(2, 50),
					),
				)
			),
			[dict(
				cohirf_kwargs=dict(
					base_model_kwargs=dict(
						eps=0.5,
						min_samples=5,
					),
				)
			)],
		),
		"BatchCoHiRF-DBSCAN-1iter": (
			BatchCoHiRF,
			dict(
				cohirf_model=BaseCoHiRF,
				cohirf_kwargs=dict(base_model=DBSCAN, max_iter=1, n_features=4, repetitions=1),
			),
			dict(
				cohirf_kwargs=dict(
					base_model_kwargs=dict(
						eps=optuna.distributions.FloatDistribution(1e-1, 10),
						min_samples=optuna.distributions.IntDistribution(2, 50),
					),
				)
			),
			[dict(
				cohirf_kwargs=dict(
					base_model_kwargs=dict(
						eps=0.5,
						min_samples=5,
					),
				)
			)],
		),
	}
)


def generate_spherical_clusters(mean_r, std, num_points_per_sphere=1000, seed=None):
	"""
	Generate 3D data points uniformly distributed on concentric spheres with labels.

	Parameters:
		mean_r (list): List of mean radii for each sphere.
		std (float): Standard deviation for the radii.
		num_points_per_sphere (int): Number of points to generate per sphere.

	Returns:
		np.ndarray: Array of shape (N, 3), where N is the total number of points.
		np.ndarray: Array of shape (N,), containing labels for each point.
	"""
	data = []
	labels = []
	generator = np.random.default_rng(seed)

	for i, r in enumerate(mean_r):
		# Generate random radii with Gaussian distribution around the mean radius
		radii = generator.normal(r, std, num_points_per_sphere)

		# Generate random points uniformly distributed on a unit sphere
		phi = generator.uniform(0, 2 * np.pi, num_points_per_sphere)
		theta = np.arccos(generator.uniform(-1, 1, num_points_per_sphere))

		# Convert spherical coordinates to Cartesian coordinates
		x = radii * np.sin(theta) * np.cos(phi)
		y = radii * np.sin(theta) * np.sin(phi)
		z = radii * np.cos(theta)

		# Stack the points and add to the data
		points = np.column_stack((x, y, z))
		data.append(points)

		# Add labels for the current sphere
		labels.extend([i] * num_points_per_sphere)

	# Combine all points and labels into single arrays
	data = np.vstack(data)
	lables = np.array(labels)
	perm = generator.permutation(data.shape[0])
	return data[perm], lables[perm]


def visualize_3d_data(data, labels):
	"""
	Visualize 3D data with colors corresponding to labels.

	Parameters:
		data (np.ndarray): Array of shape (N, 3), containing 3D points.
		labels (np.ndarray): Array of shape (N,), containing labels for each point.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")

	# Scatter plot with colors based on labels
	scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, s=2, cmap="viridis")
	plt.show()


class SphericalClusteringExperiment(ClusteringExperiment):

	def __init__(
		self,
		*args,
		n_spheres: int | list[int] = 2,
		n_samples: int | list[int] = 2000,
		radius_separation: float | list[float] = 0.5,
		radius_std: float | list[float] = 0.01,
		add_radius_as_feature: bool = False,
		seed_dataset: int | list[int] = 0,
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.n_samples = n_samples
		self.n_spheres = n_spheres
		self.radius_separation = radius_separation
		self.radius_std = radius_std
		self.add_radius_as_feature = add_radius_as_feature
		self.seed_dataset = seed_dataset

	@property
	def models_dict(self):
		return models_dict.copy()

	def _add_arguments_to_parser(self):
		super()._add_arguments_to_parser()
		if self.parser is None:
			raise ValueError("Parser is not initialized. Please provide a parser.")
		self.parser.add_argument("--n_samples", type=int, default=self.n_samples, nargs="*")
		self.parser.add_argument("--n_spheres", type=int, default=self.n_spheres)
		self.parser.add_argument("--radius_separation", type=float, default=self.radius_separation)
		self.parser.add_argument("--radius_std", type=float, default=self.radius_std)
		self.parser.add_argument("--add_radius_as_feature", action="store_true", default=self.add_radius_as_feature)
		self.parser.add_argument("--seed_dataset", type=int, default=self.seed_dataset, nargs="*")

	def _unpack_parser(self):
		args = super()._unpack_parser()
		self.n_samples = args.n_samples
		self.n_spheres = args.n_spheres
		self.radius_separation = args.radius_separation
		self.radius_std = args.radius_std
		self.add_radius_as_feature = args.add_radius_as_feature
		self.seed_dataset = args.seed_dataset
		return args

	def _get_combinations_names(self) -> list[str]:
		combination_names = super()._get_combinations_names()
		combination_names.extend(
			[
				"n_samples",
				"n_spheres",
				"radius_separation",
				"radius_std",
				"seed_dataset",
			]
		)
		return combination_names

	def _get_unique_params(self):
		unique_params = super()._get_unique_params()
		unique_params['add_radius_as_feature'] = self.add_radius_as_feature
		return unique_params

	def _load_data(
		self, combination: dict, unique_params: dict, extra_params: dict, mlflow_run_id: Optional[str] = None, **kwargs
	):
		n_samples = combination["n_samples"]
		seed_dataset = combination["seed_dataset"]
		n_spheres = combination["n_spheres"]
		radius_separation = combination["radius_separation"]
		radius_std = combination["radius_std"]
		add_radius_as_feature = unique_params["add_radius_as_feature"]
		# generate dataset name and directory
		dataset_name = (
			f"spherical_{n_samples}_{n_spheres}_{radius_separation}_{radius_std}_{add_radius_as_feature}_{seed_dataset}"
		)
		dataset_dir = self.work_root_dir / dataset_name
		X_file = dataset_dir / "X.npy"
		y_file = dataset_dir / "y.npy"
		# check if dataset is already saved and load it if it is
		if os.path.exists(X_file) and os.path.exists(y_file):
			X = np.load(X_file)
			y = np.load(y_file)
		else:
			X, y = generate_spherical_clusters(
				mean_r=np.linspace(radius_separation, radius_separation * n_spheres, n_spheres),
				std=radius_std,
				num_points_per_sphere=n_samples // n_spheres,
				seed=seed_dataset,
			)
			if add_radius_as_feature:
				# add radius as feature
				X = np.hstack((X, np.sum(X**2, axis=1, keepdims=True)))
			# save on work_dir for later use
			os.makedirs(dataset_dir, exist_ok=True)
			np.save(X_file, X)
			np.save(y_file, y)
		return {"X": X, "y": y, "dataset_name": dataset_name}

	def run_spherical_experiment_combination(
		self,
		model_nickname: str,
		seed_model: int = 0,
		seed_dataset: int = 0,
		model_params: Optional[dict] = None,
		fit_params: Optional[dict] = None,
		n_samples: int = 100,
		n_spheres: int = 2,
		radius_separation: float = 0.5,
		radius_std: float = 0.1,
		add_radius_as_feature: bool = False,
		n_jobs: int = 1,
		return_results: bool = True,
		log_to_mlflow: bool = False,
		timeout_combination: Optional[int] = None,
		timeout_fit: Optional[int] = None,
	):
		combination = {
			"model_nickname": model_nickname,
			"seed_model": seed_model,
			"seed_dataset": seed_dataset,
			"model_params": model_params,
			"fit_params": fit_params,
			"n_samples": n_samples,
		}
		unique_params = {
			"n_spheres": n_spheres,
			"radius_separation": radius_separation,
			"radius_std": radius_std,
			"add_radius_as_feature": add_radius_as_feature,
		}
		extra_params = {
			"n_jobs": n_jobs,
			"return_results": return_results,
			"timeout_combination": timeout_combination,
			"timeout_fit": timeout_fit,
		}
		if log_to_mlflow:
			return self._run_mlflow_and_train_model(
				combination=combination,
				unique_params=unique_params,
				extra_params=extra_params,
				return_results=return_results,
			)
		else:
			return self._train_model(
				combination=combination,
				unique_params=unique_params,
				extra_params=extra_params,
				return_results=return_results,
			)


if __name__ == "__main__":
	experiment = SphericalClusteringExperiment()
	experiment.run_from_cli()
