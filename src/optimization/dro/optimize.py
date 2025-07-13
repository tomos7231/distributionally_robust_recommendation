import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.optimization.dro.padm import penalty_alternating_direction_method


def optimize_dro(
    rating_df: pd.DataFrame,
    cov_matrix: np.ndarray,
    N: int,
    kappa1: float,
    kappa2: float,
    gamma: float,
    eps_inner: float,
    eps_outer: float,
    scale_gamma: float,
) -> dict[int, list[int]]:
    items_recommended = dict()

    time_list = []

    for user in tqdm(rating_df["user_id"].unique()):
        # user_idがuserでdata_typeがtestのデータを取得
        test_df = rating_df.query(f"user_id == {user} and data_type == 'test'").reset_index(drop=True)

        # もし候補商品が50個以上なら予測評価値が高い順に50個に絞る
        if len(test_df) > 50:
            test_df = test_df.sort_values("rating", ascending=False).head(50)

        # item_idを取得
        item_ids = test_df["item_id"].values
        # item_idとidxの辞書を作成
        idx2item = dict(enumerate(item_ids))

        # ratingを取得
        mu_hat = test_df["rating"].values
        # 共分散行列を取得
        Sigma_hat = cov_matrix[item_ids][:, item_ids]

        # zの初期値
        z_init = np.zeros(len(mu_hat))

        # mu_hatの値が大きい上位10個のindex
        top_indices = np.argsort(mu_hat)[-10:][::-1]

        z_init[top_indices] = N / 10

        # 交代方向乗数法
        start = time.time()
        x_opt, z_opt = penalty_alternating_direction_method(
            mu_hat, Sigma_hat, z_init, kappa1, kappa2, gamma, N, eps_inner, eps_outer, scale_gamma
        )
        elapsed_time = time.time() - start
        time_list.append(elapsed_time)

        # 結果を保存
        items_recommended[user] = [idx2item[i] for i, z in enumerate(z_opt) if z == 1]

    return items_recommended, np.mean(time_list), np.std(time_list)
