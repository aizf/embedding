import networkx as nx
from networkx.algorithms import community
from functools import reduce
from networkx.algorithms.community import greedy_modularity_communities
import json
import os

path = "./output/centrality.json"


def average_degree(G):
    total = 0
    for node in nx.degree(G):
        total += node[1]
    return total / nx.number_of_nodes(G)


def network_centrality(network):
    if os.path.exists(path):
        print("load " + path)
        with open(path, encoding="utf8") as f:
            res = json.load(f)
            return res
    else:
        print("centralityPrecessing")
    f = open("./dataset/node_level/facebook/facebook_static.json", encoding="utf8")
    network = json.load(f)
    nodes = network["nodes"]
    links = network["links"]
    isDirected = False
    if isDirected:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    # G.add_nodes_from([node["name"] for node in nodes])
    G.add_nodes_from([node["id"] for node in nodes])
    for link in links:
        source = link["source"]
        target = link["target"]
        try:
            G.edges[source, target]["weight"] += 1
        except:
            G.add_edge(source, target, weight=1)

    for n in G:
        G.nodes[n]["name"] = n
    # d = nx.readwrite.json_graph.node_link_data(G)
    # json.dump(d, open("./temp.json", "w"))
    # for n in G:
    #     print(type(n))
    # print(nx)
    print(nx.info(G), "\n")
    # print("density", nx.density(G), "\n")
    # print("degree", nx.degree(G), "\n")
    # print("cliques")
    # cliques = nx.find_cliques(G)
    # print(cliques)
    # for c in cliques:
    #     print(c)

    # print(list(community.k_clique_communities(G, 5)))
    # print("sigma", nx.sigma(G), "\n")
    # print("omega", nx.omega(G), "\n")
    # print("average_degree", average_degree(G), "\n")
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=3000)
    pagerank = nx.pagerank(G)

    data = {
        "average_degree":
            average_degree(G),
        "density":
            nx.density(G),
        "column": [
            "degree_centrality", "closeness_centrality",
            "betweenness_centrality", "eigenvector_centrality", "pagerank"
        ],
        "row": [],
        "data": []
    }

    for n in G:
        data["row"].append(n)
        data["data"].append([
            degree_centrality[n], closeness_centrality[n],
            betweenness_centrality[n], eigenvector_centrality[n], pagerank[n]
        ])
    with open(path, 'w', encoding="utf8") as f:
        json.dump(data, f)
    return data, G


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    network = json.load(open("./test/miserables.json"))
    # network = json.load(open("./test/PolBooks.json"))
    # network = json.load(open("./test/epinions_1_percent.json"))
    # network = json.load(open("./test/epinions_5_percent.json"))
    # network = json.load(open("./test/starwars-full-mentions.json"))
    data, G = network_centrality(network)
    # print(data)
    print(G.edges["Zephine", "Listolier"])

    # plt.figure(figsize=(25, 25))
    # plt.subplot(331)
    # nx.draw(G, with_labels=True)
    # plt.subplot(332)
    # nx.draw_circular(G, with_labels=True)
    # plt.subplot(333)
    # nx.draw_kamada_kawai(G, with_labels=True)
    # # plt.subplot(334)
    # # nx.draw_planar(G, with_labels=True)
    # plt.subplot(335)
    # nx.draw_random(G, with_labels=True)
    # plt.subplot(336)
    # nx.draw_spectral(G, with_labels=True)
    # plt.subplot(337)
    # nx.draw_spring(G, with_labels=True)
    # plt.subplot(338)
    # nx.draw_shell(G, with_labels=True)

    plt.show()
