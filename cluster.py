import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth, AgglomerativeClustering, AffinityPropagation
from sklearn.datasets import make_blobs

n_clusters = 4
bandwidth = None
# bandwidth = estimate_bandwidth(X, quantile=0.2)
eps = 180
linkage = ["ward", "complete", "average", "single"]
algorithms = [("KMeans", KMeans(n_clusters=n_clusters)),
              ("MeanShift", MeanShift(bandwidth=bandwidth)),
              ("DBSCAN", DBSCAN(eps=eps, min_samples=6)),
              ("AgglomerativeClustering",
               AgglomerativeClustering(linkage="ward", n_clusters=n_clusters))]


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
    import joblib
    import os
    import json

    X = np.array([])
    X_path = "./output/X.pkl"
    if os.path.exists(X_path):
        print("load " + X_path)
        X = joblib.load(X_path)
    else:
        print("error")
    print(X.shape)

    with open('./output/pos_tsne.json', encoding="utf8") as f:
        pos = np.array(json.load(f))

    plt.figure(figsize=(18, 12))
    pX = 2
    pY = 3
    names = [
        "KMeans", "MeanShift", "DBSCAN", "AgglomerativeClustering"
    ]
    # names = [
    #     "KMeans"
    # ]

    res_path = "./output/cluster_32/"
    for i, algorithm in enumerate(names):
        path = res_path + algorithm + ".json"
        if os.path.exists(path):
            print("load " + path)
            with open(path, encoding="utf8") as f:
                res = json.load(f)
        else:
            print("calc " + algorithm)
            pred = cluster(algorithm, X)
            res = pred.tolist()
            with open(path, "w", encoding="utf8") as f:
                json.dump(res, f)
        # print(res)

        plt.subplot(pX, pY, i + 1)
        plt.scatter(pos[:, 0], pos[:, 1], c=res, alpha=0.5, s=5)
        plt.title(algorithm)

    with open('./output/target_list.json', encoding="utf8") as f:
        res = np.array(json.load(f))
        plt.subplot(pX, pY, pX * pY)
        plt.scatter(pos[:, 0], pos[:, 1], c=res, alpha=0.5, s=5)
        plt.title("target")
    plt.show()
