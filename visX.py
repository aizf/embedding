from flask import Flask, request
import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

X = np.array([])
X_path = "./output/X.pkl"
if os.path.exists(X_path):
    print("load " + X_path)
    X = joblib.load(X_path)
else:
    print("error")

print(X.shape)

# tsne
x = np.array([])
x_path = "./output/x_tsne.pkl"
if os.path.exists(x_path):
    print("load " + x_path)
    x = joblib.load(x_path)
else:
    model = TSNE(n_components=2, init='pca', random_state=0, n_jobs=4)
    x = model.fit_transform(X)
    joblib.dump(x, x_path)
print(x.shape)

# umap
x = np.array([])
x_path = "./output/x_umap.pkl"
if os.path.exists(x_path):
    print("load " + x_path)
    x = joblib.load(x_path)
else:
    model = umap.UMAP(n_components=2)
    x = model.fit_transform(X)
    joblib.dump(x, x_path)
print(x.shape)

# pca
x = np.array([])
x_path = "./output/x_pca.pkl"
if os.path.exists(x_path):
    print("load " + x_path)
    x = joblib.load(x_path)
else:
    model = PCA(n_components=2)
    x = model.fit_transform(X)
    joblib.dump(x, x_path)
print(x.shape)

# isomap
x = np.array([])
x_path = "./output/x_isomap.pkl"
if os.path.exists(x_path):
    print("load " + x_path)
    x = joblib.load(x_path)
else:
    model = Isomap(n_components=2, n_jobs=4)
    x = model.fit_transform(X)
    joblib.dump(x, x_path)
print(x.shape)

# mds 爆内存
# x = np.array([])
# x_path = "./output/x_mds.pkl"
# if os.path.exists(x_path):
#     print("load " + x_path)
#     x = joblib.load(x_path)
# else:
#     model = MDS(n_components=2)
#     x = model.fit_transform(X)
#     joblib.dump(x, x_path)
# print(x.shape)

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
_x = [xx * 30 for xx in x[:, 0]]
_y = [xx * 60 for xx in x[:, 1]]
plt.scatter(_x, _y, c=c)
plt.show()

pos = []
for i in range(len(_x)):
    pos.append([_x[i], _y[i]])

with open("./output/pos_mds.json", "w", encoding='utf-8') as f:
    json.dump(pos, f)
