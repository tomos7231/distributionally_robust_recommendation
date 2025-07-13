from typing import TypeVar

import numpy as np
import pandas as pd

from src.prediction.model import (
    ItemKNNRecommender,
    NMFRecommender,
    SVDRecommender,
    UserKNNRecommender,
)

logger = TypeVar("logger")


def predict_ratings(
    cfg: dict, train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logger, model_name: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if model_name == "MF":
        model = SVDRecommender(
            train_df,
            test_df,
            logger,
            cfg.prediction.n_factors,
            cfg.prediction.n_epochs,
            cfg.prediction.lr_all,
            cfg.prediction.reg_all,
        )
    elif model_name == "NMF":
        model = NMFRecommender(
            train_df,
            test_df,
            logger,
            cfg.prediction.n_factors,
            cfg.prediction.n_epochs,
        )
    elif model_name == "USERCF":
        model = UserKNNRecommender(train_df, test_df, logger, cfg.prediction.k)
    elif model_name == "ITEMCF":
        model = ItemKNNRecommender(train_df, test_df, logger, cfg.prediction.k)
    else:
        raise Exception("Unknown model name: {}".format(model_name))

    all_pred_df, item_embs = model.run()

    return all_pred_df, item_embs
