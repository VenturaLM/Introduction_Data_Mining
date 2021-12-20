import click
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

# Iris dataset:
from sklearn.datasets import load_iris

# Additional files:
from data_management import getData
from pre_processing import stadardizeData


@click.command()
@click.option('--dataset', '-d', default=None, required=False, help=u'Name of the file with data.')
def main(dataset):
    if dataset == None:
        iris = load_iris()
        X_train, y_train = iris.data, iris.target
    else:
        X_train, y_train = getData(dataset)

    # TODO: Do we have to standardize the encoded categorical data?
    X_train_standardized = stadardizeData(X_train)

    # Compute decision tree:
    #getDecisionTree(X_train_standardized, y_train)

    # Compute K-Nearest Neighbor:
    getKNearestNeighbors(X_train_standardized, y_train)


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
    print(f'\nDecision Trees:')
    print(
        f'\tTrain score = {cross_val_score(clf, X_train, y_train, cv=3)}')

    # Plot the tree.
    tree.plot_tree(clf)
    plt.show()

    computePCADecomposition(X_train)


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
    print(f'\nK-Nearest Neighbors:')
    print(
        f'\tTrain score = {cross_val_score(neigh, X_train, y_train, cv=3)}')


def computePCADecomposition(X_train):
    """
    Principal component analysis (PCA). Reduce data dimensionality to 'n' components.
        See: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

    Parameters
    ----------
    X_train: ndarray
        Standardized data values.

    Returns
    -------

    """

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_train)

    # Percentage of variance explained by each of the selected components.
    print(f'\tExplained variance ratio = {pca.explained_variance_ratio_}')

    # TODO: Fix scatter plot colors.
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])

    plt.title("PCA components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
