import pandas as pd

from src.paths import DATA_DIR

from .preprocessor import DataPreprocessor


def make_data(
    dataset_name: str, test_size: float, min_count_rating: int, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "movielens":
        df = pd.read_csv(DATA_DIR / "u.data", names=["user_id", "item_id", "rating", "timestamp"], sep="\t")
        df = df.drop("timestamp", axis=1)
        if max(df["user_id"]) == 1:
            df["user_id"] -= 1
        if max(df["item_id"]) == 1:
            df["item_id"] -= 1
    elif dataset_name == "r3":
        df = pd.read_csv(DATA_DIR / "r3_preprocessed.csv")
    elif dataset_name == "book":
        df = pd.read_csv(DATA_DIR / "book_1214_fordro.csv")
    else:
        raise ValueError("Invalid dataset_name.")

    preprocessor = DataPreprocessor(df, test_size, min_count_rating, seed)

    filtered_df = preprocessor.filter_data(df, min_count_rating)

    train_df, test_df = preprocessor.split_data(filtered_df, test_size, dataset_name, seed)

    return train_df, test_df
