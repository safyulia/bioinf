
import pandas as pd
import numpy as np

from sklearn.datasets import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("BRCA_pam50.tsv.txt", sep="\t", index_col=0)
X = df.iloc[:, :-1].to_numpy()
y = df["Subtype"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=27)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",  NearestCentroid())
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracies = cross_val_score(
    model, X, y,
    scoring=make_scorer(accuracy_score),
    cv=RepeatedStratifiedKFold(n_repeats=100)
)
print('общая точность',np.mean(accuracies))

def class_accuracy (y_pred, y_test, Subtype):
    a=[]
    b=[]
    for i in range (len(y_test)):
        if y_test[i]==Subtype:
            b.append(y_pred[i])
            a.append(y_test[i])
    print (Subtype)
    print ('количество образцов', len (a))
    return np.sum(np.array(a)==np.array(b))/len(a)

word = ['Normal-like', 'Luminal A', 'Luminal B','Triple-negative','HER2-enriched','Healthy']
for i in range (len(word)):
    print (class_accuracy (y_pred, y_test, word[i]))