#!	/usr/bin/python
"""

Install wittgenstein 0.3.2:
$ pip install wittgenstein
Source: https://pypi.org/project/wittgenstein/

"""

import pandas as pd
from sklearn import datasets
import wittgenstein as lw
from sklearn.model_selection import train_test_split

# Read dataset
df = pd.read_csv("iris.csv")
train, test = train_test_split(df, test_size=.33)

# Create and train model
ripper_clf = lw.RIPPER()
ripper_clf.fit(df, class_feat="variety", pos_class='Versicolor')

# Print model
print(ripper_clf.out_model())

# Score
X_test = test.drop('variety', axis=1)
y_test = test['variety']
print(ripper_clf.score(X_test, y_test))

