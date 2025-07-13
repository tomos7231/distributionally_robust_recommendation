from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class ShrinkageEstimator(metaclass=ABCMeta):
    def __init__(self, delta: float, path_data: Path, definite_positive: bool = True):
        self.delta = delta
        self.path_data = path_data
        self.definite_positive = definite_positive

    def run(self) -> pd.DataFrame:
        # データの読み込み
        self.pred_rating_df = self.load_data(self.path_data)
        # 共分散行列の計算
        S, F = self.make_covariance_matrix()
        # 出現行列の計算
        freq_matrix = self.create_freq_matrix(self.pred_rating_df)
        # 共分散行列の推定
        cov_matrix = self.calc_weighted_sum(S, F, self.delta)
        return cov_matrix, freq_matrix

    @staticmethod
    def load_data(path_data: Path) -> pd.DataFrame:
        pred_rating_df = pd.read_csv(path_data / "pred_rating.csv")
        return pred_rating_df

    @staticmethod
    def create_freq_matrix(df: pd.DataFrame) -> np.ndarray:
        # あるアイテムを評価したユーザー件数とあるアイテムの組み合わせを評価したユーザー件数が格納された行列を作成
        all_item_ids = range(df["item_id"].max() + 1)
        train_df = df[df["data_type"] == "train"].reset_index(drop=True)
        # ピボットテーブルを作成
        matrix_user_item = train_df.pivot_table(index="user_id", columns="item_id", aggfunc="size", fill_value=0)
        # item_idが連続していない場合は、全てのitem_idを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=0)
        # 行列の積を計算
        item_matrix = matrix_user_item.T.dot(matrix_user_item)

        return item_matrix.values

    @abstractmethod
    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def calc_weighted_sum(S: np.ndarray, F: np.ndarray, delta: float) -> np.ndarray:
        # SとFを重み付けした共分散行列を計算
        return ((1 - delta) * S) + (delta * F)


class DiagonalEstimator(ShrinkageEstimator):
    def __init__(self, delta: float, path_data: Path, definite_positive: bool = True):
        super().__init__(delta, path_data, definite_positive)

    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        # 全てのアイテムのid
        all_item_ids = range(self.pred_rating_df["item_id"].max() + 1)

        # data_type=trainだけを抽出
        train_df = self.pred_rating_df[self.pred_rating_df["data_type"] == "train"].reset_index(drop=True)

        # user*itemの行列を作成
        matrix_user_item = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
        # trainだけのデータだとitemidが不連続になるので、全てのitemidを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=np.nan)

        # 共分散行列の計算
        S = matrix_user_item.cov().values
        # 欠損値を0にする
        S = np.nan_to_num(S, nan=0.0)
        # Sの対角成分のみを取り出す
        F = np.diag(np.diag(S))

        return S, F


class InputationEstimator(ShrinkageEstimator):
    def __init__(self, delta: float, path_data: Path, definite_positive: bool = True):
        super().__init__(delta, path_data, definite_positive)

    def make_covariance_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        # 全てのアイテムのid
        all_item_ids = range(self.pred_rating_df["item_id"].max() + 1)

        # data_type=trainだけを抽出
        train_df = self.pred_rating_df.query("data_type == 'train'").reset_index(drop=True)

        # user*itemの行列を作成
        matrix_user_item = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
        # trainだけのデータだとitemidが不連続になるので、全てのitemidを含むようにする
        matrix_user_item = matrix_user_item.reindex(columns=all_item_ids, fill_value=np.nan)

        # 共分散行列の計算
        S = matrix_user_item.cov().values

        if self.definite_positive:
            # 対角成分は0.1, 非対角成分は0で欠損値を埋める
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    if i == j:
                        S[i, j] = 0.0 if np.isnan(S[i, j]) else S[i, j]
                    else:
                        S[i, j] = 0.0 if np.isnan(S[i, j]) else S[i, j]

            # Sの固有値が負のものを正にして行列を再構成
            eig_vals, eig_vecs = np.linalg.eig(S)
            eig_vals[eig_vals < 0] = 1e-5
            S = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
            S = np.real(S)
        else:
            # 欠損値を0にする
            S = np.nan_to_num(S, nan=0.0)

        # 全てのデータで共分散行列を計算
        matrix_user_item_all = self.pred_rating_df.pivot_table(index="user_id", columns="item_id", values="rating")

        F = matrix_user_item_all.cov().values
        F = np.real(F)

        return S, F
