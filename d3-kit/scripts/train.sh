#!/usr/bin/env bash
# train.sh — 训练（4bit backbone + LoRA）
# 用法:
#   bash scripts/train.sh                        # 训练最新 L4 .train.l4.jsonl
#   bash scripts/train.sh --dry-run              # 5 步格式验证
#   bash scripts/train.sh --data l4_data/xx.l4.jsonl --max-steps 500
# 前置: validate_data.sh 已通过；训练占 GPU ~7GB，先停同卡上的推理服务
set -euo pipefail
cd "$(dirname "$0")/.." && source ./env.sh

DRY=0; DATA=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --data) DATA="$2"; shift ;;
    --max-steps) export D3_MAX_STEPS="$2"; shift ;;
    --lora-r) export D3_LORA_R="$2"; shift ;;
    --grad-accum) export D3_GRAD_ACCUM="$2"; shift ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
  shift
done
[[ -n "$DATA" ]] || DATA=$(ls -t "$D3_WORK"/l4_data/*.train.l4.jsonl 2>/dev/null | head -1 || true)
[[ -n "$DATA" && -f "$DATA" ]] || { echo "[train] FAIL: 没有 L4 训练数据，先跑 validate_data.sh"; exit 1; }
DATA="${DATA#"$D3_WORK"/}"

VER="${VER:-train-$(date +%m%d%H%M)}"
OUT="$D3_WORK/output/$VER"
mkdir -p "$OUT"
export PYTHONPATH="$D3_KIT${PYTHONPATH:+:$PYTHONPATH}"
export D3_QUANT_CONFIG=4bit HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH="$PATH:/usr/lib/wsl/lib" 2>/dev/null || true

echo "[train] 补丁自检..."
"$D3_PY" "$D3_KIT/patch/patch_d3_cache.py" --model-dir "$D3_MODEL_DIR"

STEPS=5
[[ $DRY == 0 ]] && STEPS="$D3_MAX_STEPS"
echo "[train] 模型: $D3_MODEL_DIR"
echo "[train] 数据: $DATA | steps=$STEPS lora_r=$D3_LORA_R grad_accum=$D3_GRAD_ACCUM lr=$D3_LR"
echo "[train] 产物: $OUT （每 50 步 checkpoint，train.log 实时日志）"

"$D3_PY" "$D3_KIT/scripts/train_d3.py" \
  --data "$DATA" --base "$D3_MODEL_DIR" --out "$OUT" \
  --max_steps "$STEPS" --bs 1 --grad_accum "$D3_GRAD_ACCUM" \
  --lora_r "$D3_LORA_R" --lr "$D3_LR" --save_steps 50 2>&1 | tee "$OUT/run.log"

grep -q "\[save final\]" "$OUT/run.log" || { echo "[train] FAIL: 未正常保存"; exit 1; }
echo "[train] 完成: $OUT/adapter-final"
[[ $DRY == 1 ]] && echo "[train] （dry-run 模式，仅验证数据格式）"
