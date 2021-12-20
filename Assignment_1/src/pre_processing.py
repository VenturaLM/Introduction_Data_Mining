from sklearn import preprocessing
# Sklearn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html


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
