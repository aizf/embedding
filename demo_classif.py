from karateclub.dataset import GraphReader

reader = GraphReader("twitch")

graph = reader.get_graph()
y = reader.get_target()

from karateclub import Diff2Vec, TENE
import numpy as np

model = Diff2Vec(diffusion_number=2, diffusion_cover=20, dimensions=16)
# model.fit(graph, T=np.array([]))
model.fit(graph)
X = model.get_embedding()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

downstream_model = LogisticRegression(random_state=0).fit(X_train, y_train)
y_hat = downstream_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_hat)
print('AUC: {:.4f}'.format(auc))