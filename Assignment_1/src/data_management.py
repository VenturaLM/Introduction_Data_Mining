import pandas as pd
import plotly.express as px
from pre_processing import encodeCategoricalFeatures
from scipy.io import arff


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
        if type(first_row[i]) == str or type(first_row[i]) == bytes:
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
        if dataset.endswith('.csv'):
            # Load '.csv' file.
            data = pd.read_csv(dataset, header=0, sep=';')
            df = pd.DataFrame(data)
        elif dataset.endswith('.arff'):
            # Load '.arff' file.
            data = arff.loadarff(dataset)
            df = pd.DataFrame(data[0])
        else:
            print('ERROR: file format not supported.')
    else:
        # If the is no dataset, load iris.
        print('Loading iris dataset.')
        df = px.data.iris()

    # WARNING: Checks if categorical, but only with the first row.
    # If necessary, introduce by hand the row that does not have missing values or any inconvenience.
    categorical_indexes = getCategoricalColumns(df.iloc[0])

    # Encode categorical values.
    if categorical_indexes != []:
        # Encode categorical features with the indexes of the columns.
        df = encodeCategoricalFeatures(df, categorical_indexes)

    # WARNING: Have to pay attention which is the target!
    X_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values

    return X_train, y_train, df
