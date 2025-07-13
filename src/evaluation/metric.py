import numpy as np
import pandas as pd
from tqdm import tqdm


def calc_precision(items_recommended: dict, test_df: pd.DataFrame, N: int, thres_rating: float) -> float:
    """
    Precisionを計算する関数
    """
    precisions = list()

    for user in tqdm(items_recommended.keys()):
        num_correct = 0

        # ユーザーごとにデータを抽出
        user_df = test_df[test_df["user_id"] == user].reset_index(drop=True)
        # item_idと評価値の辞書
        rating_dict = dict(zip(user_df["item_id"], user_df["rating"]))

        # 推薦したアイテムのid
        item_ids = items_recommended[user]

        # user_dfでitem_idを探し、存在するかつ評価が閾値以上の場合はnum_correctを+1する
        for item_id in item_ids:
            if item_id in rating_dict.keys() and rating_dict[item_id] >= thres_rating:
                num_correct += 1

        precisions.append(num_correct / N)

    return np.mean(precisions)


def calc_recall(items_recommended: dict, test_df: pd.DataFrame, thres_rating: float) -> float:
    """
    Recallを計算する関数
    """
    recalls = list()

    for user in tqdm(items_recommended.keys()):
        num_correct = 0

        # ユーザーごとに評価が閾値以上のアイテムを抽出
        user_df = test_df[(test_df["user_id"] == user) & (test_df["rating"] >= thres_rating)].reset_index(drop=True)
        # もしuser_dfが空の場合はrecallは0
        if user_df.shape[0] == 0:
            recalls.append(0)
        else:
            # user_dfのitem_idがどれだけ推薦したアイテムに含まれているかを計算
            num_correct = len(set(user_df["item_id"]).intersection(set(items_recommended[user])))

            recalls.append(num_correct / len(user_df))

    return np.mean(recalls)


def calc_var_hit_item(items_recommended: dict, test_df: pd.DataFrame, thres_rating: float) -> float:
    """
    利用者ごとのヒットアイテムの分散を計算する関数
    """
    hit_items = list()

    for user in tqdm(items_recommended.keys()):
        num_correct = 0

        # ユーザーごとに評価が閾値以上のアイテムを抽出
        user_df = test_df[(test_df["user_id"] == user) & (test_df["rating"] >= thres_rating)].reset_index(drop=True)
        if user_df.shape[0] == 0:
            hit_items.append(0)
        else:
            # user_dfのitem_idがどれだけ推薦したアイテムに含まれているかを計算
            num_correct = len(set(user_df["item_id"]).intersection(set(items_recommended[user])))

            hit_items.append(num_correct)

    return np.var(hit_items)


def calc_diversity(items_recommended: dict) -> int:
    """
    多様性を計算する関数
    """
    # items_recommendedに含まれる全てのitem_idの種類数
    id_recommended = set()

    for user in items_recommended.keys():
        id_recommended = id_recommended.union(set(items_recommended[user]))

    return len(id_recommended)


def calc_var_recommended(item_recommended: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    アイテムごとに推薦された回数の分散を計算する関数
    """
    # max_item_idを取得
    max_item_id = max(train_df["item_id"].max(), test_df["item_id"].max())

    # item_idをキー、デフォルト0の辞書を作成
    item_recommended_count = {i: 0 for i in range(max_item_id + 1)}

    # item_recommended_countに推薦された回数をカウント
    for user in item_recommended.keys():
        for item_id in item_recommended[user]:
            item_recommended_count[item_id] += 1

    return np.var(list(item_recommended_count.values()))


def calc_gini(item_recommended: dict, train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Gini係数を計算する関数
    """
    # max_item_idを取得
    max_item_id = max(train_df["item_id"].max(), test_df["item_id"].max())

    # item_idをキー、デフォルト0の辞書を作成
    item_recommended_count = {i: 0 for i in range(max_item_id + 1)}

    # item_recommended_countに推薦された回数をカウント
    for user in item_recommended.keys():
        for item_id in item_recommended[user]:
            item_recommended_count[item_id] += 1

    # item_recommended_countを昇順にソート
    item_recommended_count = sorted(item_recommended_count.items(), key=lambda x: x[1])

    # Gini係数を計算
    n = len(item_recommended_count)
    gini = 0
    for i in range(n):
        gini += (2 * (i + 1) - n - 1) * item_recommended_count[i][1]

    return gini / n / sum(item_recommended_count[i][1] for i in range(n))
