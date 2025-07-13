#!/bin/bash

# 任意のkappa1とkappa2の組み合わせをリストとして定義
combinations=(
    "0.5 0.5"
    "1.0 0.5"
    "2.0 0.5"
    "0.5 1.0"
    "1.0 1.0"
    "2.0 1.0"
    "0.5 2.0"
    "1.0 2.0"
    "2.0 2.0"
    "4.0 0.5"
    "4.0 1.0"
    "4.0 2.0"
    # 必要に応じて他の組み合わせも追加
)

# seedの値のリスト（42から51まで）
seed_values=(42 43 44 45 46 47 48 49 50 51)

# すべての組み合わせで実行
for seed in "${seed_values[@]}"; do
    for combo in "${combinations[@]}"; do
        # 組み合わせを分解してkappa1とkappa2に代入
        set -- $combo
        kappa1=$1
        kappa2=$2

        echo "Running experiment with seed=$seed, kappa1=$kappa1 and kappa2=$kappa2"
        uv run python main_dro.py optimization.kappa1=$kappa1 optimization.kappa2=$kappa2 seed=$seed name=article_r3_cov${seed}
    done
done