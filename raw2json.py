import numpy as np
import pandas as pd

# z = np.array([[1, 2], [1, 2]])
# a = np.array([[1, 2], [1, 2], [1, 2]])
# b = np.array([[1, 1, 1], [1, 1, 1]])
# print(a.dot(b))
# print(np.linalg.pinv(a))
nodes = []
with open("./dataset/node_level/facebook/target_raw.csv", encoding="utf8") as f:
    t = pd.read_csv(f,
                    encoding="utf8",
                    sep=","
                    )
    v = t.values
    for node in v:
        nodes.append({"id": int(node[0]), "facebook_id": str(node[1]), "page_name": node[2], "page_type": node[3]})

path = "./dataset/node_level/facebook/edges.csv"
links = []
with open(path, encoding="utf8") as f:
    t = pd.read_csv(f,
                    encoding="utf8",
                    sep=","
                    )
    v = t.values
    for link in v:
        if link[0] == link[1]: continue
        links.append({"source": int(link[0]), "target": int(link[1])})
import json

with open("./dataset/node_level/facebook/facebook.json", "w", encoding="utf8") as f:
    json.dump({"nodes": nodes, "links": links}, f)
