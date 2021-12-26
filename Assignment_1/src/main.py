import click

# Additional files:
from data_management import getData
from pre_processing import imputeMissingValues, stadardizeData
from pre_processing import sampleWithoutReplacement
from supervised_learning import getDecisionTree, getKNearestNeighbors
from dimensionality_reduction import computePCADecomposition


@click.command()
@click.option('--dataset', '-d', default=None, type=str, required=False, show_default=True, help=u'Name of the file with data.')
@click.option('--sample', '-s', default=None, required=False, show_default=True, help=u'Indicates the percentage of random samples of the dataset.')
@click.option('--tree', '-t', is_flag=True, default=False, required=False, show_default=True, help=u'Boolean that indicates the compute of a decision tree over the data.')
@click.option('--knn', '-k', is_flag=True, default=False, required=False, show_default=True, help=u'Boolean that indicates the compute of a k-nearest neighbor over the data.')
@click.option('--pca', '-p', default=2, type=int, required=False, show_default=True, help=u'Indicates the value for the dimensions of the PCA components.')
def main(dataset, sample, tree, knn, pca):

    # Load the data from a file.
    X_train, y_train, df = getData(dataset)

    # Imputation of missing values.
    #X_train = imputeMissingValues(X_train)

    # Standardize the data.
    X_train_standardized = stadardizeData(X_train)

    # Random sample of 'sample' instances without replacement over decision tree and/or k-nearest neighbor.
    if sample is not None:
        X_train_standardized, y_train = sampleWithoutReplacement(
            X_train_standardized, y_train, sample)

    # Compute decision tree.
    if tree:
        getDecisionTree(X_train_standardized, y_train)

    # Compute K-Nearest Neighbor.
    if knn:
        getKNearestNeighbors(X_train_standardized, y_train)

    # Compute principal components.
    computePCADecomposition(df, X_train, pca)


if __name__ == "__main__":
    main()
