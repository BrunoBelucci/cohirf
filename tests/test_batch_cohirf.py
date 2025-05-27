import numpy as np
from sklearn.metrics import adjusted_rand_score
from cohirf.models.batch_cohirf import BatchCoHiRF

def test_batch_cohirf_fit_predict_hierarchy_strategies():
    # Create a simple synthetic dataset
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=0, scale=1, size=(10, 5)),
        rng.normal(loc=5, scale=1, size=(10, 5)),
    ])

    # Test with hierarchy_strategy = 'parents'
    model_parents = BatchCoHiRF(
        cohirf_kwargs=dict(random_state=42),
        hierarchy_strategy="parents",
        batch_size=10,
        max_epochs=10,
        verbose=False,
    )
    labels_parents = model_parents.fit_predict(X)

    # Test with hierarchy_strategy = 'labels'
    model_labels = BatchCoHiRF(
        cohirf_kwargs=dict(random_state=42),
        hierarchy_strategy="labels",
        batch_size=10,
        max_epochs=10,
        verbose=False,
    )
    labels_labels = model_labels.fit_predict(X)

    # The labels may be permuted, so we check that the clustering assignments are equivalent up to permutation
    score = adjusted_rand_score(labels_parents, labels_labels)
    assert score == 1.0, f"ARI between strategies is {score}, expected 1.0 (identical clustering)"
