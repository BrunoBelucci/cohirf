import numpy as np
import matplotlib.pyplot as plt

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
