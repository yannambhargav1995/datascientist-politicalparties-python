import pandas as pd


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.feature_columns = data.columns

    ##### YOUR CODE GOES HERE #####

    def dim_reducer(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components)
        df = pca.fit_transform(self.data)
        return pd.DataFrame(df,columns=['pca_component_1','pca_component_2']) ,pca