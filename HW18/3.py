import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV

import math

df = pd.read_csv("sgn.csv")

x = df['x'].to_numpy()
y = df['y'].to_numpy()

m = 1000

X = []
for i in x:
    new_row = []
    for j in range(1, m + 1):
        new_row.append(math.sin(i * j))
        new_row.append(math.cos(i * j))
    X.append(new_row)
X = pd.DataFrame(X).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

model = LassoCV(positive=True)
model.fit(X_train, y_train)

print('train score:', model.score(X_train, y_train))
print('test score:', model.score(X_test, y_test))

print('coef:', model.coef_)
print('intercept:', model.intercept_)

y_pred = model.predict(X)

sns.scatterplot(x=x, y=y)
sns.lineplot(x=x, y=y_pred)
plt.ylim([-3, 3])

plt.tight_layout()
plt.savefig("test_lasso.pdf")
