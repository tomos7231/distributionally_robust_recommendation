import cvxpy as cp
import numpy as np
from cvxpy.error import SolverError


def solve_upper_problem(
    mu_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    z: np.ndarray,
    num_item: int,
    kappa1: float,
    kappa2: float,
    gamma: float,
) -> np.ndarray:
    x = cp.Variable(num_item, nonneg=True)
    # 実数値の対称行列
    P = cp.Variable((num_item, num_item), symmetric=True)
    Q = cp.Variable((num_item, num_item), symmetric=True)
    # 双対変数
    p = cp.Variable(num_item)
    q = cp.Variable(num_item)
    r = cp.Variable()
    s = cp.Variable()

    # 目的関数
    objective = cp.Minimize(
        cp.trace(((kappa2 * Sigma_hat) - mu_hat.reshape(-1, 1) * mu_hat.reshape(-1, 1).T) @ Q)
        + r
        + cp.trace(Sigma_hat @ P)
        - 2 * (mu_hat.T @ p)
        + kappa1 * s
        + gamma * cp.norm(x - z, 1)
        # PとQのフロベニウスノルムをペナルティとして追加
        + 10 * (cp.norm(P, 'fro'))
        + 10 * (cp.norm(Q, 'fro'))
    )

    # 制約条件
    constraints = [
        p == -q / 2 - Q @ mu_hat,
        cp.bmat(
            [
                [Q, cp.reshape(q / 2 + x / 2, shape=(num_item, 1), order="C")],
                [cp.reshape(q / 2 + x / 2, shape=(num_item, 1), order="C").T, cp.reshape(r, shape=(1, 1), order="C")],
            ]
        )
        >> 0,
        cp.bmat(
            [
                [P, cp.reshape(p, shape=(num_item, 1), order="C")],
                [cp.reshape(p, shape=(num_item, 1), order="C").T, cp.reshape(s, shape=(1, 1), order="C")],
            ]
        )
        >> 0,
        x <= 1,
        # p <= 10000,2
        # p >= -10000,
        # q <= 10000,
        # q >= -10000,
    ]

    # 問題を解く
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.MOSEK)
    except SolverError:
        print("MOSEK solver failed, trying SCS...")
        problem.solve(solver=cp.SCS)

    # ソルバーのステータスを確認
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Initial solver attempt failed with status: {problem.status}")
        print("Retrying with regularization terms...")

        # 正則化項を追加して再度問題を定義
        objective_with_reg = cp.Minimize(
            cp.trace(((kappa2 * Sigma_hat) - mu_hat.reshape(-1, 1) * mu_hat.reshape(-1, 1).T) @ Q)
            + r
            + cp.trace(Sigma_hat @ P)
            - 2 * (mu_hat.T @ p)
            + kappa1 * s
            + gamma * cp.norm(x - z, 1)
            + 100 * (cp.norm(P, "fro"))
            + 100 * (cp.norm(Q, "fro"))
        )

        # 正則化項付きで問題を再構築して解く
        problem_reg = cp.Problem(objective_with_reg, constraints)
        try:
            problem_reg.solve(solver=cp.MOSEK)
        except SolverError:
            print("MOSEK solver failed with regularization, trying SCS...")
            problem_reg.solve(solver=cp.SCS)

        if problem_reg.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Solver with regularization failed with status: {problem_reg.status}")
            print(f"Objective value: {problem_reg.value}")
            raise ValueError(f"Failed to solve upper problem even with regularization: {problem_reg.status}")

        problem = problem_reg

    if x.value is None:
        raise ValueError("Solver returned None for decision variable x")

    return x.value


def solve_lower_problem(x: np.ndarray, N: int) -> np.ndarray:
    # xの値が大きいN個を1, それ以外を0にした行列を返す
    threshold = np.sort(x)[::-1][N - 1]

    return (x >= threshold).astype(int)
