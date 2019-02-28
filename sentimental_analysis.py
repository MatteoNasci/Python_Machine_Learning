
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

X = [
    'odio #3tee mi sta sui coglioni',
    'tua madre è meglio di #3tee',
    'amo #3tee mi sta da dio',
    'porco dio #3tee mi sta sui coglioni',
    'ho regalato #3tee a tua madre ieri sera',
    '#3tee mi è apparso con dio e gli agnellini in sogno',
    'non è vero che #3tee fa schifo',
    '#3tee fa schifo',
    '#3tee non fa schifo',
    'ho regalato #3tee a mia madre la amo',
]

y = [
    'no', 'no', 'si', 'no', 'no', 'si', 'si', 'no', 'si', 'si',
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

print(X)
print(vectorizer.vocabulary_)

model = BernoulliNB()
model.fit(X, y)
p = model.predict(X)

print('Model accuracy: ', accuracy_score(y, p))
print('Probability: ')
print(model.predict_proba(X))
