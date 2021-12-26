import os
# https://plotly.com/graphing-libraries/
import plotly.express as px

from sklearn.decomposition import PCA


def computePCADecomposition(df, X_train, dimensions):
    """
    Principal component analysis (PCA). Reduce data dimensionality to 'n' components. Afterwards, a '.png' image is exported with a scatter plot.
        See: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

    Parameters
    ----------
    X_train: ndarray
        Standardized data values.

    Returns
    -------
    Nothing
    """

    pca = PCA(n_components=dimensions)
    principalComponents = pca.fit_transform(X_train)

    # Percentage of variance explained by each of the selected components.
    print(f'\tExplained variance ratio = {pca.explained_variance_ratio_}')

    features = input('\nPCA:\n\tSelect column for categorization:\n\t> ')

    # Check if feature exists in the dataset.
    try:
        df[features]
    except:
        print("ERROR: Could not compute PCA (selected columns does not exists in the dataset).")
        return

    # features with iris dataset must be equal to: 'species'.
    fig = px.scatter(principalComponents, x=0, y=1, color=df[features])

    # If 'images' directory does not exists --> Create it.
    if not os.path.exists("images"):
        os.mkdir("images")

    fig.write_image("images/pca.png")
