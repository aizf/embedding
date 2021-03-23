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
X_path = "./output/X_ASNE.pkl"
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

# "tsne" "umap" "pca" "isomap"
name = "tsne"
force_update = True
# force_update = False
x_path = "./output/x_{}.pkl".format(name)
x = np.array([])
if os.path.exists(x_path) and not force_update:
    print("load " + x_path)
    x = joblib.load(x_path)
else:
    model = models(name)
    x = model.fit_transform(X)
    joblib.dump(x, x_path)
print(x.shape)

_x = [xx * 20 for xx in x[:, 0]]
_y = [xx * 20 for xx in x[:, 1]]
plt.figure(figsize=(12, 18))
plt.scatter(_x, _y, c=c, alpha=0.5, s=5)
plt.show()

pos = []
for i in range(len(_x)):
    pos.append([_x[i], _y[i]])

with open("./output/pos_{}_asne.json".format(name), "w", encoding='utf-8') as f:
    json.dump(pos, f)
