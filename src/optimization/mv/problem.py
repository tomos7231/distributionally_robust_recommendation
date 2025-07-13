from __future__ import annotations

import numpy as np
from gurobipy import GRB, GurobiError, Model, quicksum


def model_optimize(
    I: list[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    freq: np.ndarray,
    alpha: float,
    N: int,
):
    """
    最適化問題を解く関数
    """
    # モデルの定義
    model = Model("mean_val_optimization")

    # INFO loggerを無効化
    model.setParam("LogToConsole", 0)
    # ログはコンソールには出すが、ファイルには出さない
    model.setParam("OutputFlag", 0)
    # 最適化時間の制限を1秒に設定
    model.setParam("TimeLimit", 4.0)

    # 変数の定義
    w = dict()
    for i in I:
        w[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"w_{i}")

    # 目的関数の定義
    objective = 0
    objective += (1 - alpha) * (quicksum(mu[i] * w[i] for i in I))

    objective -= alpha * (quicksum(sigma[i, j] * w[i] * w[j] for i in I for j in I))
    model.setObjective(objective, GRB.MAXIMIZE)

    # 制約条件の定義
    # wの和はN
    model.addConstr(quicksum(w[i] for i in I) == N, name="w_sum")

    # 最適化の実行
    try:
        model.optimize()
    except GurobiError:
        print("Error reported during optimization")

    # 結果の取得
    w_opt = np.array([w[i].x for i in I])  # どのアイテムが推薦されるか
    obj_val = model.objVal  # 目的関数値

    return w_opt, obj_val
