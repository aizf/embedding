from textPrecessing import textPrecessing
import os
import joblib

precess_text = './input/precess_text.txt'
tf_path = './input/tf.pkl'


# lda_path = './input/lda.pkl'


def lda(text, n_topics=10, n_top_words=10):
    t = text
    # if os.path.exists(precess_text):
    #     print("load " + precess_text)
    #     t = open(precess_text, 'r', encoding='utf-8').read()
    # else:
    #     print("textPrecessing")
    #     t = textPrecessing(text)
    #     with open(precess_text, 'w', encoding='utf-8') as f:
    #         f.write(t)
    # print(t)
    # 3.统计词频
    print("统计词频")
    tf_vectorizer = {}
    tf = {}
    if os.path.exists(tf_path):
        print("load " + tf_path)
        tf_vectorizer = joblib.load(tf_path)
        tf = tf_vectorizer.transform(t.split(" "))
    else:
        print("tfPrecessing")
        from sklearn.feature_extraction.text import CountVectorizer
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=1500,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(t.split(" "))
        joblib.dump(tf_vectorizer, tf_path)
    # print(tf)

    print("LDA")
    lda_path = './input/lda' + str(n_topics) + ".pkl"
    lda = {}
    if os.path.exists(lda_path):
        print("load " + lda_path)
        lda = joblib.load(lda_path)
    else:
        print("ldaPrecessing")
        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=n_topics,  # 主题数
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)  # tf即为Document_word Sparse Matrix
        joblib.dump(lda, lda_path)
        print("lda.perplexity", lda.perplexity(tf))  # 收敛效果

    def print_top_words(model, feature_names, n_top_words):
        # 打印每个主题下权重较高的term
        import numpy as np
        norm = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
        for topic_idx, topic in enumerate(norm):
            print("Topic #%d:" % topic_idx)
            print([(feature_names[i], topic[i])
                   for i in topic.argsort()[:-n_top_words - 1:-1]])
        # 打印主题-词语分布矩阵
        # print(model.components_)
        # print(model.components_.shape)

    def to_json(model, feature_names, n_top_words):
        # 打印每个主题下权重较高的term
        res = []
        import numpy as np
        norm = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
        for topic_idx, topic in enumerate(norm):
            one_topic = [(feature_names[i], topic[i])
                         for i in topic.argsort()[:-n_top_words - 1:-1]]
            res.append(one_topic)
        import json
        # with open("./output/lda.json", "w", encoding="utf8") as f:
        #     json.dump(res, f)
        return json.dumps(res)

    tf_feature_names = tf_vectorizer.get_feature_names()
    # print("tf_feature_names", tf_feature_names)
    return to_json(lda, tf_feature_names, n_top_words)
    # print_top_words(lda, tf_feature_names, n_top_words)

    # return t


if __name__ == '__main__':
    print(lda(""))
