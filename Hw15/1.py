
import pandas as pd
import numpy as np

from sklearn.datasets import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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
    ("clf", KNeighborsClassifier(n_neighbors=1, weights="distance", p=2))
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_pred, y_test))

accuracies = cross_val_score(
    model, X, y,
    scoring=make_scorer(accuracy_score),
    cv=RepeatedStratifiedKFold(n_repeats=100)
)
print(np.mean(accuracies), np.std(accuracies))
params = {
    "clf__n_neighbors": [1, 3, 5, 7],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2]
}

cv = GridSearchCV(
    model, params,
    scoring=make_scorer(accuracy_score),
    cv=RepeatedStratifiedKFold(n_repeats=10)
)
cv.fit(X, y)
print(cv.best_params_)
