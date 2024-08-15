import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame, test_size: float, min_count_rating: int, seed: int = 42):
        self.df = df
        self.seed = seed
        self.test_size = test_size
        self.min_count_rating = min_count_rating

    @staticmethod
    def filter_data(df: pd.DataFrame, min_count_rating: int) -> pd.DataFrame:
        # 評価値の数が多いユーザーのみを抽出
        user_counts = df["user_id"].value_counts()
        user_counts = user_counts[user_counts >= min_count_rating].index.tolist()
        df = df[df["user_id"].isin(user_counts)].reset_index(drop=True)

        return df

    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float, dataset_name: str, seed: int) -> pd.DataFrame:
        if dataset_name == "movielens":
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["user_id"])
            return train_df, test_df
