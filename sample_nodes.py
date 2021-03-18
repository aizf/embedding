import json
import random

path = "./dataset/node_level/facebook/facebook_static.json"
with open(path, encoding="utf8")as f:
    g = json.load(f)
nodes = g["nodes"]  # 22470
links = g["links"]  # 170823

# _nodes = random.sample(nodes, 2100)
_nodes = nodes
_links = random.sample(links, 1100)

path1 = "./dataset/node_level/facebook/facebook_static2.json"
_g = {"nodes": _nodes, "links": _links}
with open(path1, 'w', encoding="utf8")as f:
    json.dump(_g, f)
# print(len(nodes))
# print(len(links))
