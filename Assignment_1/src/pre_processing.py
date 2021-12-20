from sklearn import preprocessing


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
