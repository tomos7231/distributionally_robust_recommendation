from pathlib import Path

import numpy as np

from src.optimization.estimator import DiagonalEstimator, InputationEstimator


def make_covariance_matrix(
    path_data: Path, delta: float, estimator: str, definite_positive: bool
) -> tuple[np.ndarray, np.ndarray]:
    if estimator == "DIAG":
        estimator = DiagonalEstimator(delta, path_data, definite_positive)
    elif estimator == "INPUTE":
        estimator = InputationEstimator(delta, path_data, definite_positive)
    else:
        raise Exception("Unknown estimator name: {}".format(estimator))

    cov_matrix, freq_matrix = estimator.run()

    return cov_matrix, freq_matrix
