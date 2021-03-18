import networkx as nx
import pandas as pd
from karateclub import DeepWalk, TENE, Diff2Vec,TADW
from tene import TENE
from data_reader import GraphReader
from deepwalk import DeepWalk

# ["wikipedia 11631 170918", "twitch 7126 35324", "github 37700 289003",
# "facebook 22470 171002", "lastfm 7624 27806", "deezer 28281 92752"]
reader = GraphReader("twitch")

g = graph = reader.get_graph()
T = features = reader.get_features()
y = target = reader.get_target()

print("graph", nx.info(g))

tene = TENE(dimensions=16, alpha=0.5, beta=1e+3)
tene.fit(graph, T)
X = tene.get_embedding()

print("X", X.shape)
print(X)

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LogisticRegression(random_state=0).fit(X_train, y_train)
model = svm.SVC(probability=True)
ovr = OneVsRestClassifier(model)  # 时间更短，准确度较低
# ovo = OneVsOneClassifier(model)  # 准确度更高一点，时间更长
ovr.fit(X_train, y_train)
score = ovr.score(X_test, y_test)
print("score :", score)

# y_hat = ovr.predict_proba(X_test)[:, 1]
# auc = roc_auc_score(y_test, y_hat)
# print('AUC: {:.4f}'.format(auc))
