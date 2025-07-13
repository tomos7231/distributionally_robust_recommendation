import numpy as np

from src.optimization.dro.problem import solve_lower_problem, solve_upper_problem


def alternating_optimize(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    z_init: np.ndarray,
    kappa1: float,
    kappa2: float,
    gamma: float,
    N: int,
    eps_inner: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    # アイテム数
    num_item = len(mu_hat)
    # xの初期値(使わないが収束判定のために保持)
    x_init = np.zeros(num_item)

    # 初期値
    x_old = x_init
    z_old = z_init

    num_iter = 0

    # 外側で永遠に繰り返す
    while True:
        num_iter += 1
        # 上側問題を解く
        try:
            x_new = solve_upper_problem(mu_hat, Sigma_hat, z_old, num_item, kappa1, kappa2, gamma)
        except ValueError as e:
            print(f"Error in iteration {num_iter}: {e}")
            print(f"Current gamma: {gamma}")
            print(f"z_old shape: {z_old.shape}, values: {z_old[:10]}...")  # 最初の10要素を表示
            raise

        # 下側問題を解いてzを更新
        try:
            z_new = solve_lower_problem(x_new, N)
        except ValueError as e:
            print(f"Error in solve_lower_problem: {e}")
            print(f"x_new shape: {x_new.shape}, values: {x_new[:10]}...")
            raise

        # 収束判定（x_new - x_oldの差分の最大の絶対値がeps_inner以下なら収束）
        if np.max(np.abs(x_new - x_old)) < eps_inner:
            break
        else:
            x_old = x_new
            z_old = z_new

    return x_new, z_new, num_iter


def penalty_alternating_direction_method(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    z_init: np.ndarray,
    kappa1: float,
    kappa2: float,
    gamma: float,
    N: int,
    eps_inner: float,
    eps_outer: float,
    scale_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_old = z_init

    while True:
        x_opt, z_opt, _ = alternating_optimize(mu_hat, Sigma_hat, z_old, kappa1, kappa2, gamma, N, eps_inner)

        # 収束判定(xとzのl1ノルムの差がeps_outer以下なら収束)
        if np.linalg.norm(x_opt - z_opt, ord=1) < eps_outer:
            break
        else:
            z_old = z_opt
            gamma *= scale_gamma

    return x_opt, z_opt
