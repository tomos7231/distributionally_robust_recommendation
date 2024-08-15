import pandas as pd
from pathlib import Path

from src.paths import DATA_DIR
from .preprocessor import DataPreprocessor

def make_data(dataset_name: str, test_size: float, min_count_rating: int, path_save: Path, seed: int = 42) -> None:
    if dataset_name == "movielens":
        df = pd.read_csv(
            DATA_DIR / "u.data", names=["user_id", "item_id", "rating", "timestamp"], sep="\t"
        )
        df = df.drop("timestamp", axis=1)
        df["user_id"] -= 1
        df["item_id"] -= 1

    else:
        raise ValueError("Invalid dataset_name.")
    
    preprocessor = DataPreprocessor(df, test_size, min_count_rating, seed)

    filtered_df = preprocessor.filter_data(df, min_count_rating)

    train_df, test_df = preprocessor.split_data(filtered_df, test_size, dataset_name, seed)

    train_df.to_csv(path_save / "train.csv", index=False)
    test_df.to_csv(path_save / "test.csv", index=False)