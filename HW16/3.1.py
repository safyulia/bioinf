import pandas as pd
import numpy as np

from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve

import seaborn as sns
import matplotlib.pyplot as plt

# TPR, TNR
df = pd.read_csv("BRCA_pam50.tsv.txt", sep="\t", index_col=0)
df = df.loc[df["Subtype"].isin(["Luminal A", "Luminal B"]), ['BIRC5', 'ACTR3B', 'UBE2C', 'UBE2T', 'RRM2', 'Subtype']]

X = df.iloc[:, :-1].to_numpy()
y = df["Subtype"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel="linear")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
print(M)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)

plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, "x", c="red")
plt.savefig("roc3_2.png", dpi=300)
