import networkx as nx
import pandas as pd
import numpy as np
from karateclub import DeepWalk, TENE, Diff2Vec, Node2Vec
from data_reader import GraphReader
from deepwalk import DeepWalk
from helpers import parameter_parser, read_graph
from helpers import read_features, read_sparse_features, tab_printer
from textPrecessing import textPrecessing
import json

# g = nx.from_edgelist(pd.read_csv(edge_path).values.tolist())
target_raw_path = './dataset/node_level/facebook/target_raw.csv'
target_raw = pd.read_csv(target_raw_path).values
shape = target_raw.shape
print(shape)
# print(target_raw[:, 2].shape)
# print(target_raw
# [:, 3])
text_raw = target_raw[:, 2] + np.array([" "] * shape[0]) + target_raw[:, 3]
text_info = [textPrecessing(s) for s in text_raw]
with open('./output/text_info.json', 'w', encoding='utf8') as f:
    json.dump(text_info, f)

# print(text_info)

raw = ""
for x in text_raw:
    raw += " " + x
# print(raw)
# from LDA import lda
# print(lda(raw))

from word2vec import word2vec

# word2vec(text_raw)

# ["wikipedia 11631 170918", "twitch 7126 35324", "github 37700 289003",
# "facebook 22470 171002", "lastfm 7624 27806", "deezer 28281 92752"]
reader = GraphReader("twitch")

g = graph = reader.get_graph()
T = features = reader.get_features()
y = target = reader.get_target()

print("graph", nx.info(g))
# print(list(g.nodes))
# print(graph)
print("T.shape", T.shape)
# print(T)
print("target.shape", target.shape)
# print("target",target)

dw = Node2Vec(dimensions=16)
dw.fit(g)
X = dw.get_embedding()

# dv = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
# dv.fit(graph)
# X = dv.get_embedding()
# tene = TENE(dimensions=16, alpha=0.5, beta=1e+3)
# tene.fit(graph, T)
# X = tene.get_embedding()

print("X", X.shape)
print(X)

from classification import classification

#
classification(X, y)
