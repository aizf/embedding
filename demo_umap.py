from flask import Flask, request
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.array([])
X_path = "./output/X_SINE.pkl"
if os.path.exists(X_path):
    print("load " + X_path)
    X = joblib.load(X_path)
else:
    print("error")

print(X.shape)


def models(m):
    if m == "tsne":
        return TSNE(n_components=2, init='pca', random_state=42, n_jobs=4)
    elif m == "umap":
        import umap
        return umap.UMAP(n_components=2, n_neighbors=60, random_state=42, n_jobs=4)
    elif m == "pca":
        return PCA(n_components=2)
    elif m == "isomap":
        return Isomap(n_components=2, n_jobs=4)


# mds 爆内存
# model = MDS(n_components=2)


colors = [
    "#5B8FF9",
    "#5AD8A6",
    "#5D7092",
    "#F6BD16",
    "#E8684A",
    "#6DC8EC",
    "#9270CA",
    "#FF9D4D",
    "#269A99",
    "#BDD2FD",
    "#FF99C3",
]
target = pd.read_csv("./dataset/node_level/facebook/target.csv").values
t = target[:, 1]
c = [colors[tt] for tt in t]

plt.figure(figsize=(30, 20))
pX = 2
pY = 3
# name = "umap"
import umap

# "euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis"
# "mahalanobis","wminkowski","seuclidean","cosine","correlation","haversine"
# "hamming","jaccard","dice","russelrao","kulsinski","ll_dirichlet"
# "hellinger","rogerstanimoto","sokalmichener","sokalsneath","yule"
# 0.0, 0.1, 0.25, 0.5, 0.8, 0.99
# 2, 5, 10, 20, 50, 100, 200
# for i, num in enumerate([0.0, 0.1, 0.25, 0.5, 0.8, 0.99]):
#     try:
#         print(i)
#         model = umap.UMAP(n_components=2, min_dist=num, n_neighbors=15, metric="euclidean", random_state=42, n_jobs=4)
#         x = model.fit_transform(X)
#         plt.subplot(pX, pY, i + 1)
#         plt.scatter(x[:, 0], x[:, 1], c=c, alpha=0.5, s=5)
#         plt.title(num)
#     except:
#         continue
# plt.show()

for i, num in enumerate([10, 20, 50, 100, 200]):
    try:
        print(i)
        model = umap.UMAP(n_components=2, min_dist=0.4, n_neighbors=num, metric="euclidean", random_state=42, n_jobs=4)
        x = model.fit_transform(X)
        plt.subplot(pX, pY, i + 1)
        plt.scatter(x[:, 0], x[:, 1], c=c, alpha=0.5, s=5)
        plt.title(num)
    except:
        continue
plt.show()
