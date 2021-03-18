from flask import Flask, request
import pandas as pd
import numpy as np
from flask_cors import CORS
from lib import lda, network_centrality, communities, model, recommendNodes, cluster
import json
import joblib
import os
from data_reader import GraphReader

app = Flask(__name__)
# app.debug = True
CORS(app)

target_raw_path = './dataset/node_level/facebook/target_raw.csv'
target_raw = pd.read_csv(target_raw_path).values
shape = target_raw.shape
text_list_raw = target_raw[:, 2] + np.array([" "] * shape[0]) + target_raw[:, 3]

text_list = []
text_list_path = "./output/text_list.json"
if os.path.exists(text_list_path):
    print("load " + text_list_path)
    with open(text_list_path, 'r', encoding='utf-8') as f:
        text_list = json.load(f)
else:
    print("textPrecessing")
    from textPrecessing import textPrecessing

    text_list = [textPrecessing(text) for text in text_list_raw]
    with open(text_list_path, 'w', encoding='utf-8') as f:
        json.dump(text_list, f)

text_set = ""
for x in text_list:
    text_set += " " + " ".join(x)

# print("text_raw",text_raw)
# print("text_set", text_set)

# embed
X = np.array([])
X_path = "./output/X.pkl"
if os.path.exists(X_path):
    print("load " + X_path)
    X = joblib.load(X_path)
else:
    print("get model")
    X, M, V = model("facebook", "TSEM")
    joblib.dump(X, X_path)

print(X.shape)


@app.route('/', methods=['GET'])
def index():
    return "Hello"


@app.route('/topics', methods=['POST'])
def topics_route():
    data = request.json
    n_topics = data["n_topics"]
    n_top_words = data["n_top_words"]
    res = lda(text_set, n_topics, n_top_words)
    # print(res)
    return {'data': res}


@app.route('/network_centrality', methods=['POST'])
def network_centrality_route():
    network = request.json
    # print(data["algorithm"], data["params"])
    res, g = network_centrality(network)
    # print(res[:5], "......")
    return {'data': res}


@app.route('/communities', methods=['POST'])
def communities_route():
    # network = request.json
    # print(data["algorithm"], data["params"])
    res, g = communities({})
    # print(res[:5], "......")
    return {'data': res}


@app.route('/recommendNodes', methods=['POST'])
def recommendNodes_route():
    data = request.json
    rank = recommendNodes(data["words"], data["nodes"], data["links"], X, M, V)
    # print(res[:5], "......")
    return {'data': rank}


@app.route('/cluster', methods=['POST'])
def cluster_route():
    data = request.json
    # print(data["algorithm"], data["params"])
    pos = data["pos"]
    algorithm = data["algorithm"]
    print("algorithm: ", algorithm)
    pred = cluster(algorithm, pos)
    # print(pred)
    # res = []
    # for (i, node) in enumerate(nodes):
    #     # print("node", node)
    #     # print("pred", pred)
    #     res.append({"uid": node["uid"], "group": int(pred[i])})
    # print(res[:5], "......")
    return {'data': pred.tolist()}


# @app.route('/save', methods=['POST'])
# def save_route():
#     network = request.json
#     # print(data["algorithm"], data["params"])
#     res = save(network)
#     # print(res[:5], "......")
#     return {'data': res}


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
