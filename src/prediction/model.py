from abc import ABCMeta, abstractmethod
from typing import TypeVar

import numpy as np
import pandas as pd
from surprise import NMF, SVD, Dataset, KNNBasic, Reader, accuracy
from tqdm import tqdm

logger = TypeVar("logger")


class RecommenderSystem(metaclass=ABCMeta):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logger):
        self.train_df = train_df
        self.test_df = test_df
        self.logger = logger
        self.model = None

        # train_dfのカラムがuser_id, item_id, ratingの順番であることを保証する
        assert list(train_df.columns) == [
            "user_id",
            "item_id",
            "rating",
        ], "train_df columns must be [user_id, item_id, rating]"

    def run(self) -> pd.DataFrame:
        # データセットの作成
        self.trainset, self.testset = self.make_dataset(self.train_df, self.test_df)
        # モデルの作成と学習
        self.model = self.build_model()
        self.fit_model(self.model, self.trainset)
        # 評価
        self.evaluate(self.model, self.testset, self.logger)
        # 予測
        all_pred_df = self.predict_ratings(self.train_df, self.test_df, self.model)

        # 埋め込み行列を取得
        all_item_ids = set(self.train_df["item_id"]).union(set(self.test_df["item_id"]))
        item_embedding_dict = {item_id: self.model.qi[idx] for idx, item_id in enumerate(self.trainset.all_items())}

        # 未知のitem_idの埋め込みを平均で代替
        default_embedding = np.mean(list(item_embedding_dict.values()), axis=0)

        # test_dfのitem_idに対して埋め込みを取得
        item_embeddings = []
        for item_id in all_item_ids:
            if item_id in item_embedding_dict:
                item_embeddings.append(item_embedding_dict[item_id])
            else:
                item_embeddings.append(default_embedding)

        item_embeddings = np.array(item_embeddings)
        print(f"item_embeddings shape: {item_embeddings.shape}")
        return all_pred_df, item_embeddings

    @abstractmethod
    def build_model(self) -> None:
        raise NotImplementedError

    @staticmethod
    def make_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        max_rate = train_df["rating"].max()
        min_rate = train_df["rating"].min()
        rating_scale = (min_rate, max_rate)

        reader = Reader(rating_scale=rating_scale)
        train_surprise_data = Dataset.load_from_df(train_df, reader)
        test_surprise_data = Dataset.load_from_df(test_df, reader)

        trainset = train_surprise_data.build_full_trainset()
        testset = test_surprise_data.build_full_trainset()
        testset = testset.build_testset()

        return trainset, testset

    @staticmethod
    def fit_model(model, trainset) -> None:
        model.fit(trainset)

    @staticmethod
    def evaluate(model, testset, logger: logger) -> None:
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions)
        logger.info(f"RMSE of Test Data: {rmse}")

    @staticmethod
    def predict_ratings(train_df: pd.DataFrame, test_df: pd.DataFrame, model) -> pd.DataFrame:
        # 全ユーザーとアイテムの組み合わせを生成
        all_users = set(train_df["user_id"]).union(set(test_df["user_id"]))
        all_items = range(max(train_df["item_id"].max(), test_df["item_id"].max()) + 1)
        all_combinations = [(user, item) for user in all_users for item in all_items]

        # 結果を保存するリスト
        predictions = []

        print("Predicting ratings...")
        for user, item in tqdm(all_combinations):
            # trainsetに存在するか確認
            if not train_df[(train_df["user_id"] == user) & (train_df["item_id"] == item)].empty:
                actual_rating = train_df[(train_df["user_id"] == user) & (train_df["item_id"] == item)]["rating"].iloc[
                    0
                ]
                predictions.append([user, item, actual_rating, "train"])
            # testsetに存在するか確認
            elif not test_df[(test_df["user_id"] == user) & (test_df["item_id"] == item)].empty:
                predicted_rating = model.predict(user, item).est
                predictions.append([user, item, predicted_rating, "test"])
            else:
                # trainsetとtestsetのどちらにも存在しない場合
                predicted_rating = model.predict(user, item).est
                predictions.append([user, item, predicted_rating, "unknown"])

        predictions_df = pd.DataFrame(predictions, columns=["user_id", "item_id", "rating", "data_type"])

        return predictions_df


class SVDRecommender(RecommenderSystem):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        logger: logger,
        n_factors: int,
        n_epochs: int,
        lr_all: float,
        reg_all: float,
    ):
        super().__init__(train_df, test_df, logger)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all

    def build_model(self):
        model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42,
        )
        return model


class NMFRecommender(RecommenderSystem):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        logger: logger,
        n_factors: int,
        n_epochs: int,
    ):
        super().__init__(train_df, test_df, logger)
        self.n_factors = n_factors
        self.n_epochs = n_epochs

    def build_model(self):
        model = NMF(n_factors=self.n_factors, n_epochs=self.n_epochs, biased=True, random_state=42)
        return model


class UserKNNRecommender(RecommenderSystem):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logger, k: int):
        super().__init__(train_df, test_df, logger)
        self.k = k

    def build_model(self):
        model = KNNBasic(k=self.k, sim_options={"user_based": True})
        return model


class ItemKNNRecommender(RecommenderSystem):
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, logger: logger, k: int):
        super().__init__(train_df, test_df, logger)
        self.k = k

    def build_model(self):
        model = KNNBasic(k=self.k, sim_options={"user_based": False})
        return model
