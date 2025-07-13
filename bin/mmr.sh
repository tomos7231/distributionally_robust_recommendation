#!/bin/bash

# alphaの値のリスト
alpha_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# seedの値のリスト（42から51まで）
seed_values=(42 43 44 45 46 47 48 49 50 51)

# すべての組み合わせで実行
for seed in "${seed_values[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        echo "Running experiment with seed=$seed, alpha=$alpha"
        uv run python main_mmr.py optimization.alpha=$alpha seed=$seed name=article_r3_cov${seed}
    done
done