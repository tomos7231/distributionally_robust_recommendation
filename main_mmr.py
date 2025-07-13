import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger

from config.config import MyConfig
from src.data_processing import make_data
from src.evaluation import evaluate
from src.optimization import make_covariance_matrix, optimize_mmr
from src.prediction import predict_ratings


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: MyConfig):
    # パスを取得（hydraによって実験のdirに移動している）
    path_exp = Path(os.getcwd()) / "mmr"
    # ディレクトリの作成
    os.makedirs(path_exp, exist_ok=True)
    logger.add(os.path.join(path_exp, "main.log"))
    logger.info(f"data: {cfg.data.name}, alpha: {cfg.optimization.alpha}, N: {cfg.optimization.N}")

    # 必要なファイルのパス
    train_path = path_exp / "train.csv"
    test_path = path_exp / "test.csv"
    pred_path = path_exp / "pred_rating.csv"
    item_embs_path = path_exp / "item_embs.npy"

    # ファイルの存在チェック
    if train_path.exists() and test_path.exists() and pred_path.exists() and item_embs_path.exists():
        logger.info("All required files already exist. Skipping data generation and prediction.")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        pred_df = pd.read_csv(pred_path)
        item_embs = np.load(item_embs_path)
    else:
        logger.info("Some required files are missing. Generating data and predictions.")

        # データ生成
        train_df, test_df = make_data(
            dataset_name=cfg.data.name,
            test_size=cfg.data.test_size,
            min_count_rating=cfg.data.min_count_rating,
            seed=cfg.seed,
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # 評価値予測
        pred_df, item_embs = predict_ratings(cfg, train_df, test_df, logger, cfg.prediction.model)
        pred_df.to_csv(pred_path, index=False)
        np.save(item_embs_path, item_embs)

    # 共分散行列の作成
    cov_matrix, freq_matrix = make_covariance_matrix(
        path_exp, cfg.optimization.delta, cfg.optimization.estimator, definite_positive=True
    )
    np.save(path_exp / "cov_matrix.npy", cov_matrix)
    np.save(path_exp / "freq_matrix.npy", freq_matrix)

    # 最適化
    # 時間計測
    result, mean_time, std_time = optimize_mmr(
        cfg.optimization.alpha,
        cfg.optimization.N,
        cfg.seed,
        path_exp,
    )
    logger.info(f"mean_time: {mean_time}, std_time: {std_time}")

    # 結果の保存
    with open(path_exp / "recommended_items.pkl", "wb") as f:
        pickle.dump(result, f)

    # 評価
    f1, diversity, var_hit_item, var_recommended, gini = evaluate(
        result, logger, train_df, test_df, cfg.optimization.N, cfg.evaluation.thres_rating
    )

    # txtファイルに結果を保存
    with open(path_exp / "result.txt", "a") as f:
        # スペース区切りで保存
        f.write(
            f"{cfg.seed} {cfg.optimization.alpha} {f1} {diversity} {var_hit_item} {var_recommended} {gini} {mean_time} {std_time}\n"
        )


if __name__ == "__main__":
    main()
