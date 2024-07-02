import pandas as pd


class DensityEstimator:
    """Class to estimate Density/Distribution of the given data.
    1. Write a function to model the distribution of the political party dataset
    2. Write a function to randomly sample 10 parties from this distribution
    3. Map the randomly sampled 10 parties back to the original higher dimensional
    space as per the previously used dimensionality reduction technique.
    """

    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        self.data = data
        self.dim_reducer_model = dim_reducer
        self.feature_names = high_dim_feature_names
        self.distribution_model = None

    ##### YOUR CODE GOES HERE #####

    def distribution(self):
        from sklearn.neighbors import KernelDensity
        model =  KernelDensity(kernel='gaussian')
        self.distribution_model = model.fit(self.data)
    
    def kernal_sampling(self):
        df = self.distribution_model.sample(10)
        return pd.DataFrame(df,columns=['pca_component_1','pca_component_2'])
    
    def inverse_mapping(self):
        high_dim_features = self.dim_reducer_model.inverse_transform(self.kernal_sampling())
        return pd.DataFrame(high_dim_features,columns=self.feature_names)