import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering, AffinityPropagation
from sklearn.datasets import make_blobs

# print(data)
# print("X", X)

n_clusters = 5
bandwidth = None
# bandwidth = estimate_bandwidth(X, quantile=0.2)
eps = 180
linkage = ["ward", "complete", "average", "single"]
algorithms = [("KMeans", KMeans(n_clusters=n_clusters)),
              ("MeanShift", MeanShift(bandwidth=bandwidth)),
              ("DBSCAN", DBSCAN(eps=eps, min_samples=6)),
              ("AgglomerativeClustering",
               AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)),
              ("AffinityPropagation", AffinityPropagation())]


def getAlgorithm(_name, params, X):
    for (name, algorithm) in algorithms:
        if _name == name:
            if _name == "MeanShift":
                bandwidth = estimate_bandwidth(X, **params)
                algorithm.set_params(**{"bandwidth": bandwidth})
            else:
                algorithm.set_params(**params)
            return algorithm


def cluster(algorithm: str, pos: list, params={}):
    # print("algorithm: ", algorithm)
    # print("pos", pos)
    X = np.array(pos)
    func = getAlgorithm(algorithm, params, X)
    pred = func.fit_predict(X)
    return pred


if __name__ == "__main__":
    # import os
    # print(os.listdir("."))
    import json

    nodes = []
    with open("./test/1.json") as f:
        nodes = json.load(f)

    plt.figure(figsize=(12, 12))
    pX = 4
    pY = 2
    names = [
        "KMeans", "MeanShift", "DBSCAN", "AgglomerativeClustering",
        "AffinityPropagation"
    ]

    for i, algorithm in enumerate(names):
        pred = cluster(algorithm, nodes)
        print(pred)
        X = np.array([[d["x"], d["y"]] for d in nodes])
        plt.subplot(pX, pY, i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=pred)
        plt.title(algorithm)
    plt.show()
