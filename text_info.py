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
