from gensim.models import Word2Vec
from textPrecessing import textPrecessing
import json
import os

s_path = './input/word2vec.json'


def word2vec(textList):
    s = []
    if os.path.exists(s_path):
        print("load " + s_path)
        with open(s_path) as f:
            s = json.load(f)
    else:
        print("textPrecessing")
        s = [textPrecessing(text).split(" ") for text in textList]
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(s, f)
    print("len(s)", len(s))
    model = Word2Vec(s, min_count=1)
    y2 = model.wv.most_similar("girl", topn=10)
    # for item in y2:
    # print(item[0], item[1])
    print("-------------------------------\n")
    print("Word2Vec", len(model.wv.index2word))
    print("Word2Vec", model.wv.vectors.shape)
    # print(model.wv.vectors)
    return s
    # t_list = t.split(" ")
    # print(s)


if __name__ == '__main__':
    s = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    model = Word2Vec(s, min_count=1)
    model.similarity("say", "meow")
    print(model.wv.__getitem__(["cat", "say"]))

    y2 = model.wv.most_similar("cat", topn=10)  # 10个最相关的
    for item in y2:
        print(item[0], item[1])
    print("-------------------------------\n")
