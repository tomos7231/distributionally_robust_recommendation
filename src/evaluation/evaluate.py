from typing import TypeVar

import pandas as pd

from src.evaluation.metric import (
    calc_diversity,
    calc_gini,
    calc_precision,
    calc_recall,
    calc_var_hit_item,
    calc_var_recommended,
)

logger = TypeVar("logger")


def evaluate(
    items_recommended: dict[int, list[int]],
    logger: logger,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    N: int,
    thres_rating: float,
) -> tuple[float, float, float, float, float]:
    """
    評価指標を計算する関数
    """

    # Precisionの計算
    precision = calc_precision(items_recommended, test_df, N, thres_rating)
    logger.info(f"Precision: {precision:.5f}")

    # Recallの計算
    recall = calc_recall(items_recommended, test_df, thres_rating)
    logger.info(f"Recall: {recall:.5f}")

    # precisionとrecallからf1を計算
    f1 = 2 * (precision * recall) / (precision + recall)
    logger.info(f"F1: {f1:.5f}")

    # Diversityの計算
    diversity = calc_diversity(items_recommended)
    logger.info(f"Diversity: {diversity}")

    # ヒットアイテムの分散の計算
    var_hit_item = calc_var_hit_item(items_recommended, test_df, thres_rating)
    logger.info(f"Var of hit item: {var_hit_item}")

    # 推薦回数の分散の計算
    var_recommended = calc_var_recommended(items_recommended, train_df, test_df)
    logger.info(f"Var of recommended: {var_recommended}")

    # Gini係数の計算
    gini = calc_gini(items_recommended, train_df, test_df)
    logger.info(f"Gini: {gini}")

    return f1, diversity, var_hit_item, var_recommended, gini
