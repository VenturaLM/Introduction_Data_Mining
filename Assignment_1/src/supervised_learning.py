import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def getDecisionTree(X_train, y_train):
    """
    Computes Decision Tree multi-class classification on a dataset.

    Parameters
    ----------
    X_train: ndarray
        Values of the data (n_samples, n_features).
    y_train: ndarray
        Target.

    Returns
    -------
    Nothing.
    """

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Evaluate training score by cross-validation.
    print(f'\nDecision Tree:')
    print(
        f'\tTrain score = {cross_val_score(clf, X_train, y_train, cv=3)}')

    # Plot the tree.
    #	For better plotting try graphviz: https://scikit-learn.org/stable/modules/tree.html
    tree.plot_tree(clf)
    plt.show()


def getKNearestNeighbors(X_train, y_train):
    """
    Computes the k-nearest neighbors vote.

    Parameters
    ----------
    X_train: ndarray
        Values of the data. (n_samples, n_features).
    y_train: ndarray
        Target.
    """

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    # Evaluate training score by cross-validation.
    print(f'\nK-Nearest Neighbor:')
    print(
        f'\tTrain score = {cross_val_score(neigh, X_train, y_train, cv=3)}')
