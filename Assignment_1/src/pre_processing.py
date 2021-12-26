import numpy as np

from sklearn import preprocessing
from sklearn.utils.random import sample_without_replacement
from sklearn.impute import SimpleImputer
# Sklearn preprocessing:
# 	https://scikit-learn.org/stable/modules/preprocessing.html
# Sklearn impute missing values:
#	https://scikit-learn.org/stable/modules/impute.html


def stadardizeData(data):
    """
    Standardize data.

    Parameters
    ----------
    data: ndarray.
        Data to standardize.

    Returns
    -------
    Standardized data.
    """

    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(data)


def encodeCategoricalFeatures(df, indexes):
    """
    Encode the categorical features of a dataframe.

    Parameters
    ----------
    df: dataframe
        Dataframe to encode.
    indexes: list
        List of indexes of the categorical columns.

    Returns
    -------
    Encoded dataframe.
    """

    # Get dataframe headers.
    columns_names = df.columns.tolist()

    # Get categorical columns headers.
    cols_to_encode = [columns_names[indexes[i]] for i in range(len(indexes))]

    enc = preprocessing.OrdinalEncoder()

    # Swap unencoded columns values for encoded columns values.
    df[cols_to_encode] = enc.fit_transform(df[cols_to_encode])

    return df


def sampleWithoutReplacement(data, target, percentage):
    """
    Get a % samples of a population.

    Parameters
    ----------
    data: nparray
        Original standardized data.
    percentage:
        Percentage of desired random samples.

    Returns
    -------
    n_population * percentage random samples and targets.
    """

    n_population = len(data)

    samples = sample_without_replacement(
        n_population=n_population, n_samples=n_population*percentage).tolist()

    return data[samples], target[samples]


def imputeMissingValues(data):
    """
    Impute missing values.

    Parameters
    ----------
    data: nparray
        Original data (still not standardized).

    Returns
    -------
    Data with missing values imputed.
    """

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    return imp.fit_transform(data)
