from sklearn.feature_extraction.text import TfidfVectorizer
corpus1 = [
    'This is the first document.',
    'This document is the second document.',
   'And this is the third one.',
    'Is this the first document?',
]
corpus2 = [
   'And this is the third one.',
    'Is this the first document?'
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus1)
print(vectorizer.get_feature_names())
print(X.shape)
print(X.toarray())
