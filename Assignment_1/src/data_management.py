import pandas as pd
import plotly.express as px
from pre_processing import encodeCategoricalFeatures


def getCategoricalColumns(first_row):
    """
    Checks if the dataset has categorical values.

    Parameters
    ----------
    first_row: ndarray
        First row of the dataset.

    Returns
    -------
    List with indexex of the categorical feature columns.
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
    df: dataframe
    """

    if dataset != None:
        # Load the dataset passed by args.
        data = pd.read_csv(dataset, header=0, sep=';')
        df = pd.DataFrame(data)
    else:
        # If the is no dataset, load iris.
        df = px.data.iris()

    X_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values

    categorical_indexes = getCategoricalColumns(X_train[0])

    # Encode categorical values.
    if categorical_indexes != []:
        # Encode categorical features with the indexes of the columns.
        df = encodeCategoricalFeatures(df, categorical_indexes)
        X_train = df.iloc[:, :-1].values
        y_train = df.iloc[:, -1].values

    return X_train, y_train, df
