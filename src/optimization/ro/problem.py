import math

import numpy as np
from gurobipy import GRB, GurobiError, Model, quicksum


def solve(
    I: list[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    freq: np.ndarray,
    alpha: float,
    gamma_mu: int,
    gamma_sigma: int,
    c_mu: float,
    c_sigma: float,
    N: int,
):
    """
    最適化問題を解く関数
    """
    # モデルの定義
    model = Model("robust_optimization")

    # INFO loggerを無効化
    model.setParam("LogToConsole", 0)
    # ログはコンソールには出すが、ファイルには出さない
    model.setParam("OutputFlag", 0)
    # 最適化時間の制限を1秒に設定
    model.setParam("TimeLimit", 30.0)

    # 変数の定義
    w, p, q = dict(), dict(), dict()
    for i in I:
        w[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"w_{i}")
        p[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"p_{i}")
        for j in I:
            q[i, j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{j}")

    # 単一の変数を定義
    z = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="z")
    g = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="g")

    # 目的関数の定義
    objective = 0
    objective += (1 - alpha) * (quicksum(mu[i] * w[i] for i in I) - z * gamma_mu - quicksum(p[i] for i in I))
    objective -= alpha * (
        quicksum(sigma[i, j] * w[i] * w[j] for i in I for j in I)
        + g * gamma_sigma
        + quicksum(q[i, j] for i in I for j in I)
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # 制約条件の定義
    # wの和はN
    model.addConstr(quicksum(w[i] for i in I) == N, name="w_sum")

    # 不確実性の範囲の定義
    for i in I:
        model.addConstr(
            (c_mu * math.sqrt(sigma[i, i]) * w[i]) / math.sqrt(freq[i, i] + 1) <= z + p[i],
            name=f"mu_{i}",
        )
        for j in I:
            model.addConstr(
                (c_sigma * abs(sigma[i, j]) * w[i] * w[j]) / math.sqrt(freq[i, j] + 1) <= g + q[i, j],
                name=f"sigma_{i}_{j}",
            )

    # # 不確実性の範囲の定義
    # for i in I:
    #     model.addConstr(
    #         c_mu * math.sqrt(sigma[i, i]) * w[i] <= z + p[i],
    #         name=f"mu_{i}",
    #     )
    #     for j in I:
    #         model.addConstr(
    #             c_sigma * sigma[i, j] * w[i] * w[j] <= g + q[i, j],
    #             name=f"sigma_{i}_{j}",
    #         )

    # 最適化の実行
    try:
        model.optimize()
    except GurobiError:
        print("Error reported during optimization")

    # 結果の取得
    w_opt = np.array([w[i].x for i in I])  # どのアイテムが推薦されるか
    obj_val = model.objVal  # 目的関数値

    return w_opt, obj_val
