from pydantic.dataclasses import dataclass


@dataclass
class DataConfig:
    name: str = "movielens"
    test_size: float = 0.2
    min_count_rating: int = 50


@dataclass
class PredictionConfig:
    model: str = "MF"  # or "ITEMCF" or "MF"
    k: int = 5
    n_factors: int = 100
    n_epochs: int = 20
    lr_all: float = 0.01
    reg_all: float = 0.2


@dataclass
class OptimizationConfig:
    kappa1: float = 0.0
    kappa2: float = 1.0
    N: int = 10


@dataclass
class EvaluationConfig:
    thres_high_rating: float = 4.0


@dataclass
class MyConfig:
    data_cfg: DataConfig = DataConfig()
    train_cfg: PredictionConfig = PredictionConfig()
    optim_cfg: OptimizationConfig = OptimizationConfig()
    eval_cfg: EvaluationConfig = EvaluationConfig()
    seed: int = 42
    name: str = "default"
