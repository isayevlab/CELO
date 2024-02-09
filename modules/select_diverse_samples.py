import numpy as np
from rdkit.ML.Cluster.Butina import ClusterData as ButinaClusterData
from scipy.optimize import minimize_scalar
from sklearn.metrics import pairwise_distances


def select_init_diverse(dataset: np.array, n_sample: int, distance="euclidean"):
    pw_dist = pairwise_distances(X=dataset, metric=distance)
    pw_dist_condensed = pw_dist[np.triu_indices(pw_dist.shape[0], k=1)]

    def n_clusters_obj(cutoff):
        clusters = ButinaClusterData(data=pw_dist_condensed,
                                     nPts=pw_dist.shape[0],
                                     distThresh=cutoff,
                                     isDistData=True,
                                     reordering=True)
        abs_diff = np.abs(len(clusters) - n_sample)
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

    return centroids
