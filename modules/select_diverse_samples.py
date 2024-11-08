import numpy as np
import rdkit
from rdkit.ML.Cluster.Butina import ClusterData as ButinaClusterData
from scipy.optimize import minimize_scalar
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.decomposition import PCA


def select_init_diverse(dataset: np.array, n_sample: int, distance="euclidean", method="butina",
        reduction_method=None, n_components=2):
    if reduction_method == "umap":
        reducer = UMAP(n_neighbors=5, min_dist=0.5, metric='cosine', n_components=n_components)
        dataset = reducer.fit_transform(dataset)
    elif reduction_method == "pca":
        reducer = PCA(n_components=n_components)
        dataset = reducer.fit_transform(dataset)

    if method == "butina":
        pw_dist = pairwise_distances(X=dataset, metric=distance)
        pw_dist_condensed = pw_dist[np.triu_indices(pw_dist.shape[0], k=1)]

        def n_clusters_obj(cutoff):
            clusters = ButinaClusterData(data=pw_dist_condensed,
                                         nPts=pw_dist.shape[0],
                                         distThresh=cutoff,
                                         isDistData=True,
                                         reordering=True)
            abs_diff = max((len(clusters) - n_sample), 0) + 2 * max((n_sample - len(clusters)), 0)
            return abs_diff

        opt_result = minimize_scalar(fun=n_clusters_obj,
                                     bounds=(0, np.max(pw_dist_condensed)),
                                     method="bounded",
                                     options={"maxiter": 20, "disp": 3, 'xatol': 0.01})

        clusters = ButinaClusterData(data=pw_dist_condensed,
                                     nPts=pw_dist.shape[0],
                                     distThresh=opt_result.x,
                                     isDistData=True, reordering=True)
        centroids = [x[0] for x in clusters]

    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_sample, random_state=0).fit(dataset)
        centroids = kmeans.cluster_centers_
        closest_points = []
        for center in centroids:
            closest_point_idx = np.argmin(np.sum((dataset - center) ** 2, axis=1))
            closest_points.append(closest_point_idx)
        centroids = closest_points

    elif method == "uniform":
        centroids = np.random.choice(range(dataset.shape[0]), size=n_sample, replace=False)

    elif method == "max_min":
        pw_dist = pairwise_distances(dataset, metric=distance)
        selected_indices = [np.random.choice(range(len(dataset)))]

        for _ in range(n_sample - 1):
            dist_to_selected = pw_dist[:, selected_indices]
            min_dist_to_selected = dist_to_selected.min(axis=1)
            next_index = np.argmax(min_dist_to_selected)
            selected_indices.append(next_index)

        centroids = selected_indices

    else:
        raise ValueError(
            "Unsupported method. Choose either 'butina', 'kmeans', 'uniform', or 'max_min'.")

    return centroids