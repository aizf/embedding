import pandas as pd
import numpy as np
import json


def communities1():
    path = "./output/communities.json"
    with open(path, encoding="utf8") as f:
        old = json.load(f)
    nodes = [-1] * 22470
    for group, elms in enumerate(old):
        for el in elms:
            # print(el)
            nodes[el] = group

    print("has -1", -1 in nodes)

    path1 = "./output/communities_list.json"
    with open(path1, 'w', encoding="utf8") as f:
        json.dump(nodes, f)


def communities2():
    path = "./dataset/node_level/facebook/target.csv"
    raw = pd.read_csv(path).values
    # print(raw)
    # res = list(raw[:, -1].T.astype(int))
    res = raw[:, -1].T.tolist()
    print(res)

    print("has -1", -1 in res)

    path1 = "./output/target_list.json"
    with open(path1, 'w', encoding="utf8") as f:
        json.dump(res, f)


communities2()
