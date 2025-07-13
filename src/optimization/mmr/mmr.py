import numpy as np


def mmr(I: list[int], mu: np.ndarray, item_embs_ar: np.ndarray, alpha: float, N: int) -> list[int]:
    """
    Maximal Marginal Relevance (MMR) を使用して、リスト I から N 個のアイテムを選択します。

    Parameters:
    I (list[int]): アイテムのインデックスのリスト
    mu (np.ndarray): 各アイテムの平均ベクトル
    item_embs_ar (np.ndarray): 各アイテムの埋め込みベクトルの行列
    alpha (float): 多様性と関連性のバランスを取るためのパラメータ
    N (int): 選択するアイテムの数

    Returns:
    list[int]: 選択されたアイテムのインデックスのリスト
    """
    selected_items = []
    remaining_items = list(I.copy())

    while len(selected_items) < N:
        best_item = None
        best_score = -np.inf

        for item in remaining_items:
            relevance = mu[item]
            # 商品の埋め込みベクトルのコサイン類似度を計算
            diversity = (
                0
                if not selected_items
                else np.mean(
                    [np.dot(item_embs_ar[item], item_embs_ar[selected_item]) for selected_item in selected_items]
                )
            )
            score = alpha * relevance - (1 - alpha) * diversity

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is not None:
            selected_items.append(best_item)
            remaining_items.remove(best_item)

    return selected_items


def mmr_cov(I: list[int], mu: np.ndarray, cov_matrix: np.ndarray, alpha: float, N: int) -> list[int]:
    """
    Maximal Marginal Relevance (MMR) を使用して、リスト I から N 個のアイテムを選択します。

    Parameters:
    I (list[int]): アイテムのインデックスのリスト
    mu (np.ndarray): 各アイテムの平均ベクトル
    cov_matrix (np.ndarray): 共分散行列
    alpha (float): 多様性と関連性のバランスを取るためのパラメータ
    N (int): 選択するアイテムの数

    Returns:
    list[int]: 選択されたアイテムのインデックスのリスト
    """
    selected_items = []
    remaining_items = list(I.copy())

    while len(selected_items) < N:
        best_item = None
        best_score = -np.inf

        for item in remaining_items:
            relevance = mu[item]
            # 商品の埋め込みベクトルの共分散を計算
            diversity = (
                0
                if not selected_items
                else np.sum([cov_matrix[item, selected_item] for selected_item in selected_items])
            )
            score = alpha * relevance - (1 - alpha) * diversity

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is not None:
            selected_items.append(best_item)
            remaining_items.remove(best_item)

    return selected_items
