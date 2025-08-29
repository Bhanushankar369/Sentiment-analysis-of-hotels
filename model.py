import joblib
import pandas as pd
info = pd.read_csv('Reviews[1].xls')
df = pd.DataFrame(info)

dic = {
    'Negative': -1,
    'Neutral': 0,
    'Positive': 1
}

df['Num_label'] = df.Label.apply(lambda x: dic[x])
X = df['Review Text']
y = df.Num_label

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X, y)

# Saving the model
joblib.dump(clf, "clf.pkl")