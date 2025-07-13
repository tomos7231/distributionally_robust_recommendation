#!/bin/bash

# 任意のkappa1とkappa2の組み合わせをリストとして定義
combinations=(
    "1.0 0.1"
    "1.0 0.5"
    "1.0 1.0"
    "5.0 0.1"
    "5.0 0.5"
    "5.0 1.0"
    "10.0 0.1"
    "10.0 0.5"
    "10.0 1.0"
    # 必要に応じて他の組み合わせも追加
)

# 各組み合わせで実行
for combo in "${combinations[@]}"; do
    # 組み合わせを分解してkappa1とkappa2に代入
    set -- $combo
    kappa1=$1
    kappa2=$2

    echo "Running experiment with kappa1=$kappa1 and kappa2=$kappa2"
    rye run python3 main_dro.py optimization.kappa1=$kappa1 optimization.kappa2=$kappa2
done