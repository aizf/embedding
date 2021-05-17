import json
import os

import numpy as np
from gensim.models import Word2Vec
from .textPrecessing import textPrecessing
from nltk.stem.porter import PorterStemmer


def word2vec(textList):
    s_path = './input/word2vec.json'
    s = []
    if os.path.exists(s_path):
        print("load " + s_path)
        with open(s_path) as f:
            s = json.load(f)
    else:
        print("textPrecessing")
        s = [textPrecessing(text) for text in textList]
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(s, f)
    print("len(s)", len(s))
    model = Word2Vec(s, min_count=1)
    return model


ps = PorterStemmer()

path = "./input/dict.json"
with open(path, 'r', encoding='utf-8') as f:
    text_dict = json.load(f)

# print(word2vec)
word2vec_model = word2vec([])


def word2stem(word: str) -> str:
    return ps.stem(word.split(";")[0])


# def recommendNodes(_words, nodes, links, target, X, M, V):
#     words = [word2stem(word) for word in _words]
#     print(words)
#     vectors_T = [word2vec_model.wv[word] for word in words]
#     if len(words) <= 0:
#         avg_T = vectors_T
#     else:
#         avg_T = np.zeros_like(vectors_T[0])
#         for i, T in enumerate(vectors_T):
#             avg_T += T
#         avg_T /= len(words)
#     # print(avg_T.shape)
#     # print(type(avg_T))
#     import joblib
#     wv_vectors = joblib.load("./output/wv_vectors.pkl")
#     from lib import cos
#     sim_score = []
#     for v in wv_vectors:
#         sim_score.append(cos(v, avg_T))
#     return sim_score

def recommendNodes(X, M, V):
    id = 11090
    sim_score = []
    from lib import cos
    for x in X:
        sim_score.append(cos(X[id], x))
    return sim_score
