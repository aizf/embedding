import networkx as nx
import matplotlib.pyplot as plt
import json

gs = nx.graph_atlas_g()
print(gs)
print(len(gs))
g = gs[522]
# print(nx.to_dict_of_dicts(g))
# print(nx.to_dict_of_lists(g))
# print(nx.to_edgelist(g))
nx.draw(g)
plt.draw()
# plt.show()

jsons = []
for g in gs:
    dic = nx.to_dict_of_lists(g)
    nodes = []
    links = []
    for i in dic:
        nodes.append({"id": i})
        links += [{"source": i, "target": j} for j in dic[i]]
    jsons.append({"nodes": nodes, "links": links})
print(jsons[:5])
with open("./output/strucs.json", 'w', encoding="utf8") as f:
    json.dump(jsons, f)
