import os
from pathlib import Path

import hydra
from loguru import logger

from config.config import MyConfig
from src.data_processing import make_data


@hydra.main(version_base=None, config_path="config/", config_name="config")
def main(cfg: MyConfig):
    path_exp = Path(os.getcwd())
    logger.add(os.path.join(path_exp, "main.log"))

    # データ生成
    make_data(
        dataset_name=cfg.data.name,
        test_size=cfg.data.test_size,
        min_count_rating=cfg.data.min_count_rating,
        path_save=path_exp,
        seed=cfg.seed,
    )


if __name__ == "__main__":
    main()
