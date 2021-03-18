import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json

dic = {}


def textPrecessing(text):
    # 小写化
    text = text.lower()
    # 去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, ' ')
    # 分词
    wordLst = nltk.word_tokenize(text)
    # 去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    # 仅保留名词或特定POS
    refiltered = nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    # 词干化
    ps = PorterStemmer()
    filtered_ = []
    for w in filtered:
        w_ = ps.stem(w)
        if w_ not in dic:
            dic[w_] = set()
            dic[w_].add(w)
        else:
            dic[w_].add(w)
        filtered_.append(w_)
    filtered = filtered_
    return filtered
    # print(filtered)

    # for i in dic:
    #     dic[i] = list(dic[i])
    #
    # with open("./input/dict.json", "w", encoding="utf8") as f:
    #     json.dump(dic, f)
    #
    #     t = " ".join(filtered)
    #     return t


if __name__ == '__main__':
    t = textPrecessing('The Voice of China 中国好声音 tvshow')
    print(t)
