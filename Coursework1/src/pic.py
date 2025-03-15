import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import solve
import heapq
import math
import time

class PIC(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, k=20, a=0.95, z=0.01):
        """
        Parameters:
          n_clusters: desired number of clusters.
          k: number of nearest neighbors for graph construction.
          a: free parameter used in setting the scale s^2.
          z: weighting parameter in path integral (should be in (0,1)).
        """
        self.n_clusters = n_clusters
        self.k = k
        self.a = a
        self.z = z

    def _build_graph(self, X):
        n_samples = X.shape[0]
        # Compute 3-nearest neighbors (if available) to estimate s^2.
        nn3 = NearestNeighbors(n_neighbors=min(4, n_samples)).fit(X)
        distances_3, indices_3 = nn3.kneighbors(X)
        # Exclude self (first column)
        d3 = distances_3[:, 1:]
        avg_sq = np.mean(d3**2)
        s2 = avg_sq / (-np.log(self.a))
        
        # Use k nearest neighbors for graph construction.
        nn = NearestNeighbors(n_neighbors=min(self.k + 1, n_samples)).fit(X)
        distances, indices = nn.kneighbors(X)
        
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            # skip the first neighbor (itself)
            for j, d in zip(indices[i, 1:], distances[i, 1:]):
                W[i, j] = np.exp(- (d**2) / s2)
        # Compute transition matrix P = D^-1 W
        D = np.sum(W, axis=1)
        # Avoid division by zero
        D[D==0] = 1e-10
        P = W / D[:, None]
        return P

    def _initial_clusters(self, X):
        # Each sample is connected with its nearest neighbor (ignoring self).
        n_samples = X.shape[0]
        nn = NearestNeighbors(n_neighbors=2).fit(X)
        distances, indices = nn.kneighbors(X)
        # Create an undirected edge between i and its nearest neighbor.
        parent = np.arange(n_samples)
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        for i in range(n_samples):
            j = indices[i, 1]
            union(i, j)
        # Form clusters from connected components.
        clusters = {}
        for i in range(n_samples):
            root = find(i)
            clusters.setdefault(root, []).append(i)
        return list(clusters.values())

    def _compute_S(self, P, indices):
        # Compute the cluster's path integral S_C = (1/|C|^2)*1^T (I - z P_C)^{-1} 1.
        P_sub = P[np.ix_(indices, indices)]
        n = len(indices)
        I = np.eye(n)
        try:
            # Solve (I - z*P_sub) y = ones.
            y = solve(I - self.z * P_sub, np.ones(n))
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(I - self.z * P_sub, np.ones(n), rcond=None)[0]
        S = np.sum(y) / (n**2)
        return S

    def _compute_S_cond(self, P, cluster_indices, union_indices):
        # Compute conditional path integral:
        # S_{C|U} = (1/|C|^2)*1_C^T (I - z*P_U)^{-1} 1_C,
        # where 1_C is indicator (1 for indices in cluster, 0 otherwise) of length |U|.
        P_sub = P[np.ix_(union_indices, union_indices)]
        nU = len(union_indices)
        I = np.eye(nU)
        b = np.zeros(nU)
        # Mark ones for indices in cluster_indices (map to positions in union_indices)
        idx_map = {idx: pos for pos, idx in enumerate(union_indices)}
        for idx in cluster_indices:
            if idx in idx_map:
                b[idx_map[idx]] = 1
        try:
            y = solve(I - self.z * P_sub, b)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(I - self.z * P_sub, b, rcond=None)[0]
        S_cond = np.sum(y) / (len(cluster_indices)**2)
        return S_cond

    def _merge_clusters(self, clusters, P):
        # Priority queue (max heap, store negative affinities to simulate max-heap)
        heap = []
        
        # start_time = time.time()
        
        # Compute initial affinities for all pairs
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                Ca, Cb = clusters[i], clusters[j]
                union = sorted(set(Ca) | set(Cb))
                SA, SB = self._compute_S(P, Ca), self._compute_S(P, Cb)
                SA_cond = self._compute_S_cond(P, Ca, union)
                SB_cond = self._compute_S_cond(P, Cb, union)
                affinity = (SA_cond - SA) + (SB_cond - SB)
                heapq.heappush(heap, (-affinity, i, j))  # Max heap (store negative values)
        
        # end_time = time.time()
        # print(f"Time taken for initial affinity computation: {end_time - start_time} seconds")

        # Keep track of active clusters
        active_clusters = list(range(len(clusters)))
        last_best_affinity = None
        
        # start_time = time.time()

        while len(active_clusters) > self.n_clusters and heap:
            # Get the best pair to merge
            best_affinity, i, j = heapq.heappop(heap)
            best_affinity = -best_affinity  # Convert back to positive
            
            # Check if both clusters are still active
            if i not in active_clusters or j not in active_clusters:
                continue  # Skip this pair, as one or both clusters have been merged
            
            # Merge clusters i and j
            new_cluster = list(set(clusters[i]) | set(clusters[j]))
            clusters.append(new_cluster)
            
            # Update active clusters
            new_index = len(clusters) - 1
            active_clusters.remove(i)
            active_clusters.remove(j)
            active_clusters.append(new_index)
            
            last_best_affinity = best_affinity
            
            # Update heap with new affinities involving the merged cluster
            for k in active_clusters:
                if k == new_index:
                    continue
                Ck = clusters[k]
                union = sorted(set(Ck) | set(new_cluster))
                Sk = self._compute_S(P, Ck)
                S_new = self._compute_S(P, new_cluster)
                Sk_cond = self._compute_S_cond(P, Ck, union)
                S_new_cond = self._compute_S_cond(P, new_cluster, union)
                affinity = (Sk_cond - Sk) + (S_new_cond - S_new)
                heapq.heappush(heap, (-affinity, k, new_index))
        
        # end_time = time.time()
        # print(f"Time taken for cluster merging computation: {end_time - start_time} seconds")

        # Return only the active clusters
        result_clusters = [clusters[i] for i in active_clusters]
        
        return result_clusters, last_best_affinity


    def fit(self, X, y=None):
        X = np.asarray(X)
        self.P_ = self._build_graph(X)
        # Initial clusters from nearest neighbor merging.
        clusters = self._initial_clusters(X)
        # Merge clusters until reaching desired number.
        clusters, _ = self._merge_clusters(clusters, self.P_)
        # Create labels for training samples.
        self.labels_ = -np.ones(X.shape[0], dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label
        # For predict, we compute centroids (here using Euclidean mean).
        self.cluster_centers_ = np.array([np.mean(X[self.labels_ == l], axis=0) 
                                        for l in range(self.n_clusters)])
        return self

    def predict(self, X):
        # Assign each new sample to the nearest cluster center.
        X = np.asarray(X)
        labels = np.argmin(np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2), axis=1)
        return labels

    def fit_predict(self, X, y=None):
        start_time = time.time()
        labels = self.fit(X, y).labels_
        end_time = time.time()
        print(f"Totoal time taken for fit_predict: {end_time - start_time} seconds")
        return labels
