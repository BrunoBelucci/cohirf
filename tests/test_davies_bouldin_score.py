"""
Test suite for the custom Davies-Bouldin score implementation. 

This module tests the custom memory-efficient Davies-Bouldin score implementation
against sklearn's implementation to ensure mathematical correctness while
providing better memory efficiency for large numbers of clusters.

Warning: Mostly implemented by Copilot, but checked for correctness.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import davies_bouldin_score as sklearn_db_score
from cohirf.metrics import davies_bouldin_score as custom_db_score


class TestDaviesBouldinScore:
    """Test class for Davies-Bouldin score implementation."""

    def test_small_dataset_exact_match(self):
        """Test that custom implementation matches sklearn on small datasets."""
        # Create small test dataset
        X, y_true = make_blobs(
            n_samples=300, 
            centers=5, 
            n_features=3, 
            random_state=42,
            cluster_std=1.0
        )
        
        # Cluster the data
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Compare implementations
        sklearn_score = sklearn_db_score(X, labels)
        custom_score = custom_db_score(X, labels, chunk_size=10)
        
        # Should match within numerical precision
        assert abs(sklearn_score - custom_score) < 1e-10, (
            f"Scores don't match: sklearn={sklearn_score:.10f}, "
            f"custom={custom_score:.10f}, diff={abs(sklearn_score - custom_score):.2e}"
        )

    def test_medium_dataset_different_chunk_sizes(self):
        """Test consistency across different chunk sizes."""
        X, _ = make_blobs(
            n_samples=1000, 
            centers=15, 
            n_features=4, 
            random_state=42
        )
        
        kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Test different chunk sizes
        chunk_sizes = [5, 10, 20, 50]
        scores = []
        
        for chunk_size in chunk_sizes:
            score = custom_db_score(X, labels, chunk_size=chunk_size)
            scores.append(score)
        
        # All scores should be identical regardless of chunk size
        for i in range(1, len(scores)):
            assert abs(scores[0] - scores[i]) < 1e-10, (
                f"Inconsistent scores across chunk sizes: "
                f"chunk_size={chunk_sizes[0]} -> {scores[0]:.10f}, "
                f"chunk_size={chunk_sizes[i]} -> {scores[i]:.10f}"
            )
        
        # Also compare with sklearn
        sklearn_score = sklearn_db_score(X, labels)
        assert abs(sklearn_score - scores[0]) < 1e-10

    def test_large_number_of_clusters(self):
        """Test with many clusters to verify chunking works correctly."""
        # Create dataset with many small clusters
        n_clusters = 100
        X, _ = make_blobs(
            n_samples=2000, 
            centers=n_clusters, 
            n_features=3, 
            random_state=42,
            cluster_std=0.5
        )
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        labels = kmeans.fit_predict(X)
        
        # Test with small chunk size to ensure chunking is exercised
        custom_score = custom_db_score(X, labels, chunk_size=25)
        sklearn_score = sklearn_db_score(X, labels)
        
        assert abs(sklearn_score - custom_score) < 1e-10, (
            f"Large cluster test failed: sklearn={sklearn_score:.10f}, "
            f"custom={custom_score:.10f}"
        )

    def test_edge_case_single_cluster(self):
        """Test edge case with single cluster."""
        X = np.random.randn(100, 3)
        labels = np.zeros(100, dtype=int)  # All points in one cluster
        
        # sklearn raises ValueError for single cluster, but our implementation should handle it
        with pytest.raises(ValueError, match="Number of labels is 1"):
            sklearn_db_score(X, labels)
        
        # Our implementation should handle single cluster gracefully and return 0.0
        custom_score = custom_db_score(X, labels)
        assert abs(custom_score - 0.0) < 1e-10

    def test_edge_case_two_clusters(self):
        """Test simple case with exactly two clusters."""
        # Create two well-separated clusters
        X = np.vstack([
            np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
            np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
        ])
        labels = np.array([0] * 50 + [1] * 50)
        
        sklearn_score = sklearn_db_score(X, labels)
        custom_score = custom_db_score(X, labels, chunk_size=1)
        
        assert abs(sklearn_score - custom_score) < 1e-10

    def test_perfect_clustering(self):
        """Test with perfect clustering (each point is its own cluster)."""
        X = np.random.randn(21, 3)  # Use 21 points so we can have 20 clusters
        labels = np.arange(20)  # First 20 points each in their own cluster
        labels = np.append(labels, 19)  # Last point goes to cluster 19
        
        # Both implementations should handle this
        sklearn_score = sklearn_db_score(X, labels)
        custom_score = custom_db_score(X, labels, chunk_size=5)
        
        assert abs(sklearn_score - custom_score) < 1e-10

    def test_identical_clusters(self):
        """Test with identical cluster centers (pathological case)."""
        # Create clusters with same center but different points
        center = np.array([0, 0])
        X = np.vstack([
            center + np.random.normal(scale=0.1, size=(25, 2)),
            center + np.random.normal(scale=0.1, size=(25, 2))
        ])
        labels = np.array([0] * 25 + [1] * 25)
        
        sklearn_score = sklearn_db_score(X, labels)
        custom_score = custom_db_score(X, labels, chunk_size=1)
        
        assert abs(sklearn_score - custom_score) < 1e-8  # Slightly looser tolerance for this case

    def test_input_validation(self):
        """Test that function handles various input types correctly."""
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Test with different input types
        sklearn_score = sklearn_db_score(X, labels)
        
        # Test with lists
        custom_score1 = custom_db_score(X.tolist(), labels.tolist())
        assert abs(sklearn_score - custom_score1) < 1e-10
        
        # Test with different chunk sizes
        custom_score2 = custom_db_score(X, labels, chunk_size=1)
        custom_score3 = custom_db_score(X, labels, chunk_size=100)
        
        assert abs(custom_score2 - custom_score3) < 1e-10

    @pytest.mark.parametrize("n_samples,n_clusters,n_features", [
        (200, 5, 2),
        (500, 10, 3),
        (300, 8, 5),
        (150, 3, 4),
    ])
    def test_parametrized_datasets(self, n_samples, n_clusters, n_features):
        """Parametrized test with different dataset configurations."""
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=n_clusters,
            n_features=n_features,
            random_state=42
        )
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sklearn_score = sklearn_db_score(X, labels)
        custom_score = custom_db_score(X, labels, chunk_size=max(1, n_clusters // 3))
        
        assert abs(sklearn_score - custom_score) < 1e-10, (
            f"Failed for n_samples={n_samples}, n_clusters={n_clusters}, "
            f"n_features={n_features}: sklearn={sklearn_score:.10f}, "
            f"custom={custom_score:.10f}"
        )

    def test_memory_efficiency_simulation(self):
        """Test that demonstrates the memory efficiency benefit."""
        # This test simulates a scenario where sklearn would use more memory
        # We can't easily test actual memory usage, but we can verify correctness
        # with a moderately large number of clusters
        
        n_clusters = 200  # Would create 200x200 = 40K distance matrix in sklearn
        X, _ = make_blobs(
            n_samples=4000,
            centers=n_clusters,
            n_features=2,
            random_state=42,
            cluster_std=0.3
        )
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        labels = kmeans.fit_predict(X)
        
        # Use small chunk size to force chunking
        custom_score = custom_db_score(X, labels, chunk_size=50)
        sklearn_score = sklearn_db_score(X, labels)
        
        assert abs(sklearn_score - custom_score) < 1e-10, (
            f"Memory efficiency test failed: sklearn={sklearn_score:.10f}, "
            f"custom={custom_score:.10f}"
        )

    def test_score_properties(self):
        """Test that the score has expected mathematical properties."""
        # Create two different clustering results for the same data
        X, true_labels = make_blobs(n_samples=500, centers=5, random_state=42)
        
        # Good clustering (matches true clusters)
        kmeans_good = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels_good = kmeans_good.fit_predict(X)
        
        # Bad clustering (wrong number of clusters)
        kmeans_bad = KMeans(n_clusters=15, random_state=42, n_init=10)
        labels_bad = kmeans_bad.fit_predict(X)
        
        score_good = custom_db_score(X, labels_good)
        score_bad = custom_db_score(X, labels_bad)
        
        # Good clustering should have lower DB score than bad clustering
        assert score_good < score_bad, (
            f"Good clustering should have lower DB score: "
            f"good={score_good:.6f}, bad={score_bad:.6f}"
        )
        
        # Scores should be non-negative
        assert score_good >= 0
        assert score_bad >= 0


def test_integration_with_existing_metrics():
    """Test that the DB score works well with other metrics in the module."""
    from cohirf.metrics import adjusted_rand_score as custom_ari
    
    X, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)
    kmeans = KMeans(n_clusters=4, random_state=42)
    pred_labels = kmeans.fit_predict(X)
    
    # Both metrics should run without errors
    db_score = custom_db_score(X, pred_labels)
    ari_score = custom_ari(true_labels, pred_labels)
    
    # Basic sanity checks
    assert db_score >= 0  # DB score is non-negative
    assert -1 <= ari_score <= 1  # ARI is bounded


if __name__ == "__main__":
    # Run tests if script is executed directly
    import sys
    
    test_instance = TestDaviesBouldinScore()
    
    methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    print(f"Running {len(methods)} Davies-Bouldin score tests...")
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            print(f"  {method_name}...", end=" ")
            method = getattr(test_instance, method_name)
            method()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
    
    # Run parametrized tests manually
    print("  test_parametrized_datasets...", end=" ")
    try:
        test_configs = [
            (200, 5, 2),
            (500, 10, 3), 
            (300, 8, 5),
            (150, 3, 4),
        ]
        for config in test_configs:
            test_instance.test_parametrized_datasets(*config)
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")
        failed += 1
    
    # Run integration test
    print("  test_integration_with_existing_metrics...", end=" ")
    try:
        test_integration_with_existing_metrics()
        print("✓ PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ FAILED: {e}")
        failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
