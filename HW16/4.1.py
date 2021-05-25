from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("BRCA_pam50.tsv.txt", sep="\t", index_col=0)
X = df.iloc[:, :-1].to_numpy()
y = df.loc[:,"Subtype"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel="linear")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
print(M)
