import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

def synthetic_dataset_1(cluster_density=0.75, random_state=42):
    n_samples = int(400*cluster_density)
    
    # Generate two dense clusters
    X1, _ = make_blobs(n_samples=n_samples, centers=[[0, -0.33]], cluster_std=0.1, random_state=random_state)
    X2, _ = make_blobs(n_samples=n_samples, centers=[[0, 0.33]], cluster_std=0.1, random_state=random_state)

    # Generate circular pattern
    X_circle, _ = make_circles(n_samples=n_samples*2, factor=0.99, noise=0.02, random_state=random_state)

    # Generate sparse noise around
    np.random.seed(random_state)
    X_noise = np.random.uniform(-1.5, 1.5, (n_samples*2, 2))

    # Remove points inside the circumference centered at (0,0) with radius 1.2
    distance_from_center = np.sqrt(np.sum(X_noise**2, axis=1))
    X_noise = X_noise[distance_from_center > 1.3]

    # Assign labels to each cluster
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    y_circle = np.full(X_circle.shape[0], 2)
    y_noise = np.full(X_noise.shape[0], 3)

    # Combine all points and labels
    X = np.vstack((X1, X2, X_circle, X_noise))
    Y = np.concatenate((y1, y2, y_circle, y_noise))

    return X, Y

def synthetic_dataset_2(cluster_density=0.75, random_state=42):
    cluster_density = 0.75
    n_samples = int(400*cluster_density)

    Pi = np.pi

    # Generate two dense clusters
    X1, _ = make_blobs(n_samples=n_samples, centers=[[-3/4*Pi+0.1, 0]], cluster_std=0.2, random_state=random_state)
    X2, _ = make_blobs(n_samples=n_samples, centers=[[-1/4*Pi-0.1, 0]], cluster_std=0.2, random_state=random_state)

    # Generate sinusoidal cluster
    np.random.seed(random_state)
    X_sin_raw = np.random.uniform(-Pi, Pi, n_samples*3)
    sin_noise = np.random.normal(0, 0.05, n_samples*3)
    X_sin = np.column_stack((X_sin_raw, -np.sin(X_sin_raw)*Pi/2 + sin_noise))

    # Generate line cluster
    X_line_raw = np.random.uniform(1/5*Pi, 1/3*Pi, n_samples)
    line_noise = np.random.normal(0, 0.05, n_samples)
    X_line = np.column_stack((X_line_raw, 5 * X_line_raw - Pi) + line_noise)

    # Generate square cluster
    X_square = np.random.uniform(0, 2/5*Pi, (n_samples, 2))
    X_square[:, 0] = X_square[:, 0] + 0.4*Pi
    X_square[:, 1] = X_square[:, 1] - 0.1*Pi

    # Assign labels to each cluster
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    y_sin = np.full(X_sin.shape[0], 2)
    y_line = np.full(X_line.shape[0], 3)
    y_square = np.full(X_square.shape[0], 4)

    # Combine all points and labels
    X = np.vstack((X1, X2, X_sin, X_line, X_square))
    Y = np.concatenate((y1, y2, y_sin, y_line, y_square))

    return X, Y

def synthetic_dataset_3(cluster_density=0.75, random_state=42):
    n_samples = int(500*cluster_density)
    
    # Generate two dense clusters
    X1, _ = make_blobs(n_samples=n_samples*2, centers=[[-0.5, 0]], cluster_std=0.05, random_state=random_state)
    X2, _ = make_blobs(n_samples=n_samples*2, centers=[[0.5, 0]], cluster_std=0.05, random_state=random_state)

    # Generate sparse noise around
    np.random.seed(random_state)
    X_noise = np.random.uniform(-1, 1, (n_samples, 2))

    # Assign labels to each cluster
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    y_noise = np.full(X_noise.shape[0], 2)

    # Combine all points and labels
    X = np.vstack((X_noise, X1, X2))
    Y = np.concatenate((y_noise, y1, y2))

    return X, Y