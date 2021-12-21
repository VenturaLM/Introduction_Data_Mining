import click
import os
import matplotlib.pyplot as plt

# https://plotly.com/graphing-libraries/
import plotly.express as px

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
@click.option('--dataset', '-d', default=None, type=str, required=False, show_default=True, help=u'Name of the file with data.')
@click.option('--tree', '-t', is_flag=True, default=False, required=False, show_default=True, help=u'Boolean that indicates the compute of a decision tree over the data.')
@click.option('--knn', '-k', is_flag=True, default=False, required=False, show_default=True, help=u'Boolean that indicates the compute of a k-nearest neighbor over the data.')
@click.option('--pca', '-p', default=2, type=int, required=False, show_default=True, help=u'Indicates the value for the dimensions of the PCA components.')
def main(dataset, tree, knn, pca):

    # Get the data.
    X_train, y_train, df = getData(dataset)

    # Standardize the data.
    X_train_standardized = stadardizeData(X_train)

    # Compute decision tree.
    if tree:
        getDecisionTree(X_train_standardized, y_train)

    # Compute K-Nearest Neighbor.
    if knn:
        getKNearestNeighbors(X_train_standardized, y_train)

    # Compute principal components.
    computePCADecomposition(df, X_train, pca)


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
    print(f'\nK-Nearest Neighbors:')
    print(
        f'\tTrain score = {cross_val_score(neigh, X_train, y_train, cv=3)}')


def computePCADecomposition(df, X_train, dimensions):
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

    pca = PCA(n_components=dimensions)
    principalComponents = pca.fit_transform(X_train)

    # Percentage of variance explained by each of the selected components.
    print(f'\tExplained variance ratio = {pca.explained_variance_ratio_}')

    features = input('\nPCA:\n\tSelect column for categorization:\n\t> ')

    # features with iris dataset must be equal to: 'species'.
    fig = px.scatter(principalComponents, x=0, y=1, color=df[features])

    # If 'images' directory does not exists --> Create it.
    if not os.path.exists("images"):
        os.mkdir("images")

    fig.write_image("images/fig1.png")


if __name__ == "__main__":
    main()
