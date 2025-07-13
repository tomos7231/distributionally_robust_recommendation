#!/bin/bash
# フォーマット用スクリプト

echo "Running ruff format..."
uv run ruff format ./

echo "Running ruff check with auto-fix..."
uv run ruff check ./ --fix

echo "Format completed!"