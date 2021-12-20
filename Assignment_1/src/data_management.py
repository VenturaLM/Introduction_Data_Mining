import pandas as pd


def getCategoricalColumns(first_row):
    """
    Checks if the dataset has categorical values.

    Parameters
    ----------
    first_row: ndarray
        First row of the dataset.

    Returns
    -------

    """

    # List with the index of each categorial column.
    categorical_cols = []

    # Append the indexes of categorical columns.
    for i in range(len(first_row)):
        if type(first_row[i]) == str:
            categorical_cols.append(i)

    return categorical_cols


def getData(dataset):
    """
    Gets the data of a dataset.

    Parameters
    ----------
    dataset: String
        Path of the dataset.

    Returns
    -------
    X_train: ndarray
        Values of the data (n_samples, n_features).
    y_train: ndarray
        Target.

    """

    # Load the dataset.
    data = pd.read_csv(dataset, header=1, sep=';')
    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    categorical_indexes = getCategoricalColumns(X_train[0])
    if categorical_indexes != []:
        # TODO: Encode categorical features with the indexes of the columns:
        # https://scikit-learn.org/stable/modules/preprocessing.html
        print("CATEGORICAL")

    return X_train, y_train
