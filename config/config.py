from dataclasses import dataclass


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
    estimator: str = "DIAG"
    delta: float = 0.5
    kappa1: float = 0.0
    kappa2: float = 1.0
    N: int = 10
    eps_outer: float = 1e-5
    eps_inner: float = 1e-5
    gamma: float = 1.5
    scale_gamma: float = 10.0
    alpha: float = 0.5
    gamma_mu: int = 0
    gamma_sigma: int = 0
    c_mu: float = 0.0
    c_sigma: float = 0.0


@dataclass
class EvaluationConfig:
    thres_rating: float = 4.0


@dataclass
class MyConfig:
    data_cfg: DataConfig = DataConfig()
    train_cfg: PredictionConfig = PredictionConfig()
    optim_cfg: OptimizationConfig = OptimizationConfig()
    eval_cfg: EvaluationConfig = EvaluationConfig()
    seed: int = 42
    name: str = "default"
