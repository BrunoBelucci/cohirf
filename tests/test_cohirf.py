import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from cohirf.models.cohirf import CoHiRF

def test_fit_predict_hierarchy_strategies():
    # Create a simple synthetic dataset
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=0, scale=1, size=(10, 5)),
        rng.normal(loc=5, scale=1, size=(10, 5)),
    ])

    # Test with hierarchy_strategy = 'parents'
    model_parents = CoHiRF(
        repetitions=5,
        random_state=42,
        n_features=0.5,
        hierarchy_strategy="parents",
        max_iter=10,
    )
    labels_parents = model_parents.fit_predict(X)

    # Test with hierarchy_strategy = 'clusters'
    model_clusters = CoHiRF(
        repetitions=5,
        random_state=42,
        n_features=0.5,
        hierarchy_strategy="clusters",
        max_iter=10,
    )
    labels_clusters = model_clusters.fit_predict(X)

    # The labels may be permuted, so we check that the clustering assignments are equivalent up to permutation
    # This can be done by checking that the contingency matrix has a perfect matching
    score = adjusted_rand_score(labels_parents, labels_clusters)
    assert score == 1.0, f"ARI between strategies is {score}, expected 1.0 (identical clustering)"

def test_cohirf_blobs_high_ari():
    # Generate easy blobs
    X, y_true = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42, return_centers=False)  # type: ignore
    model = CoHiRF(
        repetitions=3,
        random_state=42,
        n_features=0.5,
        kmeans_n_clusters=2,
        max_iter=10,
    )
    y_pred = model.fit_predict(X)
    ari = adjusted_rand_score(y_true, y_pred)
    assert ari > 0.9, f"ARI is {ari}, expected > 0.9 for easy blobs"

def test_cohirf_one_feature():
    # X with only one feature
    X = np.array([[0], [1], [2], [3]])
    model = CoHiRF(
        repetitions=2,
        random_state=42,
        n_features=1.0,
        max_iter=5,
    )
    labels = model.fit_predict(X)
    assert labels.shape == (4,), f"Expected 4 labels, got {labels.shape}"
