import json

path = "./input/pos1.json"
with open(path, 'r', encoding='utf-8') as f:
    g = json.load(f)
    nodes = g["nodes"]
res = []
for node in nodes:
    res.append([node["x"], node["y"]])
path = "./output/pos_force.json"
with open(path, 'w', encoding='utf-8') as f:
    json.dump(res, f)
