import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

def clustering_error(y_true, y_pred):
    """
    Compute the clustering error (CE) after finding the best cluster-to-label assignment.
    
    Parameters:
    - y_true: Ground truth labels (array-like)
    - y_pred: Cluster assignments (array-like)
    
    Returns:
    - Clustering error (float, between 0 and 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique labels and clusters
    labels = np.unique(y_true)
    clusters = np.unique(y_pred)
    
    # Create confusion matrix
    cost_matrix = np.zeros((len(labels), len(clusters)))
    
    for i, label in enumerate(labels):
        for j, cluster in enumerate(clusters):
            cost_matrix[i, j] = np.sum((y_true == label) & (y_pred == cluster))
    
    # Solve the optimal assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    
    # Compute clustering error
    correct = cost_matrix[row_ind, col_ind].sum()
    total = len(y_true)
    
    return 1 - (correct / total)


def create_graph(X, n_neighbors=20):
    """Create a graph from the data with KNN."""
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=False)
    return connectivity.toarray()

def von_neumann_kernel(X, alpha=0.95, z=0.01):
    """Compute the von Neumann kernel."""
    n = X.shape[0]
    W = create_graph(X, n_neighbors=20)
    # Make symmetric
    W = np.maximum(W, W.T)
    
    # Create diagonal matrix D
    D = np.diag(np.sum(W, axis=1))
    
    # Normalized Laplacian
    L = np.eye(n) - np.linalg.inv(np.sqrt(D)) @ W @ np.linalg.inv(np.sqrt(D))
    
    # Compute the kernel
    K = np.linalg.inv(np.eye(n) + z * L)
    
    # Apply alpha parameter for diffusion
    K_alpha = np.linalg.matrix_power(K, int(1/(1-alpha)))
    
    return K_alpha

def k_medoids(X, n_clusters):
    """K-medoids clustering without using sklearn-extra."""
    # Calculate pairwise distances
    distances = euclidean_distances(X)
    
    # Initialize medoids randomly
    n_samples = X.shape[0]
    medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    
    # Initialize cluster assignments
    labels = np.zeros(n_samples, dtype=int)
    
    max_iterations = 100
    for _ in range(max_iterations):
        # Assign each point to closest medoid
        old_labels = labels.copy()
        for i in range(n_samples):
            distances_to_medoids = [distances[i, idx] for idx in medoid_indices]
            labels[i] = np.argmin(distances_to_medoids)
        
        # Check for convergence
        if np.array_equal(old_labels, labels):
            break
        
        # Update medoids
        for cluster_idx in range(n_clusters):
            cluster_points = np.where(labels == cluster_idx)[0]
            if len(cluster_points) > 0:
                # Find the point that minimizes total distance to other points in cluster
                intra_cluster_distances = distances[cluster_points][:, cluster_points]
                total_distances = intra_cluster_distances.sum(axis=1)
                medoid_idx_in_cluster = np.argmin(total_distances)
                medoid_indices[cluster_idx] = cluster_points[medoid_idx_in_cluster]
    
    return labels

def average_linkage(X, n_clusters):
    """Average linkage hierarchical clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    return model.fit_predict(X)

def single_linkage(X, n_clusters):
    """Single linkage hierarchical clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
    return model.fit_predict(X)

def complete_linkage(X, n_clusters):
    """Complete linkage hierarchical clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    return model.fit_predict(X)

def affinity_propagation(X):
    """Affinity Propagation clustering."""
    model = AffinityPropagation(random_state=0)
    return model.fit_predict(X)

def normalized_cuts(X, n_clusters):
    """Normalized Cuts spectral clustering."""
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                              n_neighbors=20, assign_labels='discretize')
    return model.fit_predict(X)

def njw_algorithm(X, n_clusters):
    """NJW algorithm (Spectral clustering by Ng, Jordan, and Weiss)."""
    # Create graph
    W = create_graph(X, n_neighbors=20)
    W = np.maximum(W, W.T)  # Make symmetric
    
    # Create diagonal matrix D
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    
    # Compute normalized Laplacian
    L_norm = np.eye(X.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    
    # Get eigenvectors
    eigenvals, eigenvects = eigh(L_norm, subset_by_index=[1, n_clusters])
    
    # Normalize rows
    rows_norm = np.sqrt(np.sum(eigenvects**2, axis=1)).reshape(-1, 1)
    Y = eigenvects / rows_norm
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(Y)

def commute_time_clustering(X, n_clusters):
    """Commute Time-based Clustering."""
    # Create graph
    W = create_graph(X, n_neighbors=20)
    W = np.maximum(W, W.T)  # Make symmetric
    
    # Create diagonal matrix D
    D = np.diag(np.sum(W, axis=1))
    
    # Compute pseudo-inverse of Laplacian
    L = D - W
    eigenvals, eigenvects = eigh(L)
    
    # Exclude the first eigenvalue (which should be 0 or close to 0)
    mask = eigenvals > 1e-10
    eigenvals_filtered = eigenvals[mask]
    eigenvects_filtered = eigenvects[:, mask]
    
    # Compute embedding
    commute_embedding = eigenvects_filtered @ np.diag(1.0 / np.sqrt(eigenvals_filtered))
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(commute_embedding)

def zeta_function_clustering(X, n_clusters, z=0.01):
    """Zeta Function-based Clustering."""
    # Create graph
    W = create_graph(X, n_neighbors=20)
    W = np.maximum(W, W.T)  # Make symmetric
    
    # Create diagonal matrix D
    D = np.diag(np.sum(W, axis=1))
    
    # Normalized Laplacian
    L_norm = np.eye(X.shape[0]) - np.linalg.inv(np.sqrt(D)) @ W @ np.linalg.inv(np.sqrt(D))
    
    # Compute zeta function kernel (similar to resolvent but using z/(z+λ))
    eigenvals, eigenvects = eigh(L_norm)
    zeta_kernel = eigenvects @ np.diag(z / (z + eigenvals)) @ eigenvects.T
    
    # Apply hierarchical clustering on the kernel
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return model.fit_predict(1 - zeta_kernel)  # Convert similarity to distance

def connectivity_kernel_clustering(X, n_clusters):
    """Connectivity Kernel-based Clustering."""
    # Create graph
    W = create_graph(X, n_neighbors=20)
    W = np.maximum(W, W.T)  # Make symmetric
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(W)
    
    # Compute shortest path lengths
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Create connectivity kernel matrix
    n = X.shape[0]
    connectivity_kernel = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j in path_lengths[i]:
                # Use exponential decay based on path length
                connectivity_kernel[i, j] = np.exp(-path_lengths[i][j])
            else:
                connectivity_kernel[i, j] = 0
    
    # Apply hierarchical clustering on the kernel
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return model.fit_predict(1 - connectivity_kernel)  # Convert similarity to distance

def diffusion_kernel_clustering(X, n_clusters, alpha=0.95, z=0.01):
    """Diffusion Kernel-based Clustering using von Neumann kernel."""
    # Compute the von Neumann kernel
    kernel = von_neumann_kernel(X, alpha=alpha, z=z)
    
    # Apply average linkage clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    return model.fit_predict(1 - kernel)  # Convert similarity to distance

def evaluate_clustering_algorithms(X, y_true, n_clusters, y_pred_pic):
    """Evaluate all clustering algorithms and return their NMI and CE scores."""
    
    algorithms = {
        'k-med': lambda: k_medoids(X, n_clusters),
        'A-link': lambda: average_linkage(X, n_clusters),
        'S-link': lambda: single_linkage(X, n_clusters),
        'C-link': lambda: complete_linkage(X, n_clusters),
        'AP': lambda: affinity_propagation(X),
        'NCuts': lambda: normalized_cuts(X, n_clusters),
        'NJW': lambda: njw_algorithm(X, n_clusters),
        'CT': lambda: commute_time_clustering(X, n_clusters),
        'Zell': lambda: zeta_function_clustering(X, n_clusters, z=0.01),
        'C-kernel': lambda: connectivity_kernel_clustering(X, n_clusters),
        'D-kernel': lambda: diffusion_kernel_clustering(X, n_clusters, alpha=0.95, z=0.01)
    }
    
    results = {
        'Algorithm': [],
        'NMI': [],
        'CE': [],
        'Silhouette': []
    }
    
    # Add PIC results first
    nmi_pic = normalized_mutual_info_score(y_true, y_pred_pic)
    ce_pic = clustering_error(y_true, y_pred_pic)
    silhouette_pic = silhouette_score(X, y_pred_pic)

    results['Algorithm'].append('PIC')
    results['NMI'].append(nmi_pic)
    results['CE'].append(ce_pic)
    results['Silhouette'].append(silhouette_pic)

    print(f"PIC: NMI = {nmi_pic:.4f}, CE = {ce_pic:.4f}, Silhouette = {silhouette_pic:.4f}")
    
    # Evaluate all other algorithms
    for name, algo in algorithms.items():
        # print(f"Running {name}...")
        try:
            y_pred = algo()

            nmi = normalized_mutual_info_score(y_true, y_pred)
            ce = clustering_error(y_true, y_pred)
            silhouette = silhouette_score(X, y_pred)
            
            results['Algorithm'].append(name)
            results['NMI'].append(nmi)
            results['CE'].append(ce)
            results['Silhouette'].append(silhouette)
            
            print(f"{name}: NMI = {nmi:.4f}, CE = {ce:.4f}, Silhouette = {silhouette:.4f}")
        except Exception as e:
            print(f"Error with {name}: {e}")
            results['Algorithm'].append(name)
            results['NMI'].append(np.nan)
            results['CE'].append(np.nan)
            results['Silhouette'].append(np.nan)
    
    print("\n")
    # Create and return DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def plot_clustering_algorithms(X, y_true, n_clusters, y_pred_pic, plots_path):
    """Plot all clustering algorithms and save their results in files."""
    
    algorithms = {
        'k-med': lambda: k_medoids(X, n_clusters),
        'A-link': lambda: average_linkage(X, n_clusters),
        'S-link': lambda: single_linkage(X, n_clusters),
        'C-link': lambda: complete_linkage(X, n_clusters),
        'AP': lambda: affinity_propagation(X),
        'NCuts': lambda: normalized_cuts(X, n_clusters),
        'NJW': lambda: njw_algorithm(X, n_clusters),
        'CT': lambda: commute_time_clustering(X, n_clusters),
        'Zell': lambda: zeta_function_clustering(X, n_clusters, z=0.01),
        'C-kernel': lambda: connectivity_kernel_clustering(X, n_clusters),
        'D-kernel': lambda: diffusion_kernel_clustering(X, n_clusters, alpha=0.95, z=0.01)
    }
    
    os.makedirs(plots_path, exist_ok=True)

    # Plot ground truth
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=20)
    plt.title('Ground Truth', fontsize=16)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.savefig(os.path.join(plots_path, 'GT_clustering.png'))
    plt.close()

    # Plot PIC results
    nmi_pic = normalized_mutual_info_score(y_true, y_pred_pic)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_pic, cmap='viridis', s=20)
    plt.title(f'PIC ({nmi_pic:.4f})', fontsize=16)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.savefig(os.path.join(plots_path, 'PIC_clustering.png'))
    plt.close()
    
    # Plot all other algorithms
    for name, algo in algorithms.items():
        try:
            y_pred = algo()
            nmi = normalized_mutual_info_score(y_true, y_pred)
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=20)
            plt.title(f'{name} ({nmi:.4f})', fontsize=16)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            plt.savefig(os.path.join(plots_path, f'{name}_clustering.png'))
            plt.close()
        except Exception as e:
            print(f"Error with {name}: {e}")

def plot_silhouette(X, y_pred, n_clusters, plot_path, dataset):
    """Create a compact silhouette plot for display in an article."""

    silhouette_avg = silhouette_score(X, y_pred)
    sample_silhouette_values = silhouette_samples(X, y_pred)

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=100)  # Smaller figure size
    
    # Set the axis limits
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[y_pred == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, 
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Make cluster label more compact
        if size_cluster_i > 30:  # Only add text if cluster is large enough
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=8)
        
        y_lower = y_upper + 10

    # Add average silhouette line
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=1)
    
    # Add compact title and labels
    ax1.set_title(f"Silhouette Plot ({dataset})", fontsize=10)
    ax1.set_xlabel("Silhouette coefficient", fontsize=9)
    ax1.set_ylabel("Cluster", fontsize=9)
    
    # Reduce tick size
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()  # Optimize layout
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()