from pathlib import Path
from typing import List
from urllib.request import urlretrieve

import pandas as pd


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        return df.drop_duplicates()

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        """Write a function to remove certain features cols and set certain cols as indices
        in a dataframe"""
        df = df.drop(columns=non_features)
        df.set_index(index)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to handle NaN values in a dataframe"""
        ##### YOUR CODE GOES HERE #####
        from sklearn.impute import SimpleImputer
        simple_impute = SimpleImputer(strategy='median')
        df_imputed = simple_impute.fit_transform(df)
        return pd.DataFrame(df_imputed, columns=df.columns)

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to normalise values in a dataframe. Use StandardScaler."""
        ##### YOUR CODE GOES HERE #####
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        return pd.DataFrame(df_scaled,columns=df.columns)


    def preprocess_data(self):
        """Write a function to combine all pre-processing steps for the dataset"""
        self.party_data = self.remove_duplicates(self.party_data)
        self.party_data = self.remove_nonfeature_cols(self.party_data,non_features=['eu_econ_require','eu_political_require','eu_googov_require','party','eu_foreign','eu_intmark','eu_budgets','eu_asylum','immigrate_policy','civlib_laworder','country'],index=['party_id'])
        self.party_data = self.handle_NaN_values(self.party_data)
        self.party_data = self.scale_features(self.party_data)
        return self.party_data

