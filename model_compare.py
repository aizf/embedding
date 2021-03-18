import networkx as nx
import pandas as pd
from data_reader import GraphReader
from classification import classification


def compare_attributed(g, T, y):
    from karateclub import BANE, TENE, TADW, FSCNMF, SINE, MUSAE, FeatherNode, ASNE, AE
    attributed_models = {
        # AE(), MUSAE(),
        # "FeatherNode": FeatherNode(),
        "BANE": BANE(),
        "TENE": TENE(),
        "TADW": TADW(),
        "FSCNMF": FSCNMF(),
        "SINE": SINE(),
        "ASNE": ASNE()
    }

    for name in attributed_models:
        model = attributed_models[name]
        model.fit(g, T)
        X = model.get_embedding()
        print(name, "X", X.shape)
        classification(X, y)


def compare_struc(g, T, y):
    n = len(g.nodes)
    from ge import DeepWalk, LINE, Node2Vec, SDNE, Struc2Vec
    structural_models = {
        "DeepWalk": DeepWalk(g, 10, 80, workers=4),
        "LINE": LINE(g, embedding_size=128, order='second'),
        "Node2Vec": Node2Vec(g, walk_length=10, num_walks=80, p=0.25, q=4, workers=4),
        # "SDNE": SDNE(g, hidden_size=[256, 128]),
        "Struc2Vec": Struc2Vec(g, 10, 80, workers=4)
    }
    for name in structural_models:
        model = structural_models[name]
        model.train()
        X_ = model.get_embeddings()
        X = []
        for i in range(n):
            X.append(X_[str(i)])
        print(name, "X", X.shape)
        classification(X, y)


if __name__ == '__main__':
    # ["wikipedia 11631 170918", "twitch 7126 35324", "github 37700 289003",
    # "facebook 22470 171002", "lastfm 7624 27806", "deezer 28281 92752"]
    for data in ["lastfm", "facebook", "wikipedia"]:
        print(data)
        reader = GraphReader(data)
        g = graph = reader.get_graph()
        T = features = reader.get_features()
        y = target = reader.get_target()

        # compare_attributed(g, T, y)
        compare_struc(g, T, y)
