#!/usr/bin/env bash
# validate_data.sh — 数据验证 + holdout 切分 + L3→L4 转换
# 用法: bash scripts/validate_data.sh data/你的数据.jsonl [--use-all]
set -euo pipefail
cd "$(dirname "$0")/.." && source ./env.sh

DATA="${1:?用法: validate_data.sh <L3 jsonl 路径>}"; shift || true
mkdir -p "$D3_WORK/data" "$D3_WORK/l4_data"
D3_MODEL_DIR="$D3_MODEL_DIR" "$D3_PY" "$D3_KIT/scripts/validate_data.py" --data "$DATA" "$@"
