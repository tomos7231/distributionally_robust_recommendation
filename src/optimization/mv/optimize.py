from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.optimization.mv.problem import model_optimize


def optimize_mv(alpha: float, N: int, seed: int, path_data: Path) -> dict[int, list[int]]:
    """
    ユーザーごとに最適化問題を解く関数
    """
    # データの読み込み
    pred_rating_df = pd.read_csv(path_data / "pred_rating.csv")
    sigma_ar = np.load(path_data / "cov_matrix.npy")
    freq_ar = np.load(path_data / "freq_matrix.npy")

    # ユーザーごとに最適化問題を解く
    items_recommended = dict()

    time_list = []

    for user in tqdm(pred_rating_df["user_id"].unique()):
        # ユーザーごとにデータを抽出
        user_df = pred_rating_df[pred_rating_df["user_id"] == user].reset_index(drop=True)
        # 予測評価値
        mu = user_df["rating"].values

        # Iはtestデータのitem_id
        test_df = user_df[user_df["data_type"] == "test"].reset_index(drop=True)

        # Iはraring順に並び替えたitem_idの上位n_candidate個
        I = test_df.sort_values("rating", ascending=False)["item_id"].values[:50]

        # 計測開始
        start = time.time()
        # 最適化問題を解く
        w_opt, obj_val = model_optimize(I, mu, sigma_ar, freq_ar, alpha, N)
        # 計測終了
        elapsed_time = time.time() - start
        time_list.append(elapsed_time)
        # 推薦したアイテムのid
        item_ids = I[w_opt >= 0.99]

        # 結果を格納
        items_recommended[user] = item_ids

    return items_recommended, np.mean(time_list), np.std(time_list)
