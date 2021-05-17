from .tsem import TSEM

dic = {"TSEM": TSEM}
data = "facebook"
model_name = "TSEM"


def model(data, model_name):
    from data_reader import GraphReader
    reader = GraphReader(data)
    g = graph = reader.get_graph()
    T = features = reader.get_features()
    y = target = reader.get_target()
    model = dic[model_name](dimensions=16)
    model.fit(g, T)
    X = model.get_embedding()
    return X
