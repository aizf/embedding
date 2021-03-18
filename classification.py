from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier


def classification(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # model = LogisticRegression(random_state=0).fit(X_train, y_train)
    model = svm.SVC(probability=True)
    ovr = OneVsRestClassifier(model)  # 时间更短，准确度较低
    # ovo = OneVsOneClassifier(model)  # 准确度更高一点，时间更长
    ovr.fit(X_train, y_train)
    score = ovr.score(X_test, y_test)
    print("score :", score)

    # y_hat = ovr.predict_proba(X_test)[:, 1]
    # auc = roc_auc_score(y_test, y_hat)
    # print('AUC: {:.4f}'.format(auc))
