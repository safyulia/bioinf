import pandas as pd
import numpy as np

from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_roc_curve

import seaborn as sns
import matplotlib.pyplot as plt

import time

np.random.seed(17)
N =10000
X1 = np.random.normal(loc=0, size=(N,1))
X2 = np.random.normal(loc=0, size=(N,1))
X3 = np.random.normal(loc=0, size=(N,1))
X4 = np.random.normal(loc=0, size=(N,1))

X1 = np.vstack([X1, X3])
X2 = np.vstack([X2, X4])

X= np.hstack([X1,X2])
y = np.array([0] * N + [1] * N)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel="linear")

import time
start_time = time.time()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("--- %s seconds ---" % (time.time() - start_time))

print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
print(M)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)

plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, "x", c="red")
plt.savefig("roc5_2.png", dpi=300)