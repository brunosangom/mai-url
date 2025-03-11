import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import kneighbors_graph
import networkx as nx
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
        'CE': []
    }
    
    # Add PIC results first
    nmi_pic = normalized_mutual_info_score(y_true, y_pred_pic)
    ce_pic = clustering_error(y_true, y_pred_pic)
    results['Algorithm'].append('PIC')
    results['NMI'].append(nmi_pic)
    results['CE'].append(ce_pic)
    print(f"PIC: NMI = {nmi_pic:.4f}, CE = {ce_pic:.4f}")
    
    # Evaluate all other algorithms
    for name, algo in algorithms.items():
        print(f"Running {name}...")
        try:
            y_pred = algo()
            nmi = normalized_mutual_info_score(y_true, y_pred)
            ce = clustering_error(y_true, y_pred)
            
            results['Algorithm'].append(name)
            results['NMI'].append(nmi)
            results['CE'].append(ce)
            
            print(f"{name}: NMI = {nmi:.4f}, CE = {ce:.4f}")
        except Exception as e:
            print(f"Error with {name}: {e}")
            results['Algorithm'].append(name)
            results['NMI'].append(np.nan)
            results['CE'].append(np.nan)
    
    # Create and return DataFrame
    results_df = pd.DataFrame(results)
    return results_df
