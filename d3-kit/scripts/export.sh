#!/usr/bin/env bash
# export.sh — LoRA 合并 → GGUF 导出（F16 基准 + Q8_0 部署版）
# 用法: bash scripts/export.sh <adapter目录> [版本名]
#   例: bash scripts/export.sh output/train-1018/adapter-final v2-1018
set -euo pipefail
cd "$(dirname "$0")/.." && source ./env.sh

ADAPTER="${1:?用法: export.sh <adapter目录> [版本名]}"
ADAPTER="${ADAPTER#"$D3_WORK"/}"
[[ -d "$D3_WORK/$ADAPTER" ]] && ADAPTER="$D3_WORK/$ADAPTER"
[[ -d "$ADAPTER" ]] || { echo "[export] FAIL: adapter 不存在: $ADAPTER"; exit 1; }

VER="${2:-export-$(date +%m%d%H%M)}"
MERGED="$D3_WORK/output/merged-$VER"
NAME="d3-$VER"
export D3_QUANT_CONFIG=none HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$D3_KIT${PYTHONPATH:+:$PYTHONPATH}"

echo "[export] ① LoRA 合并 -> fp16（bf16 重载底座）..."
"$D3_PY" "$D3_KIT/scripts/merge_lora_to_gguf.py" \
  --adapter "$ADAPTER" --base "$D3_MODEL_DIR" --out "$MERGED"

F16_DIR="$MERGED/merged_fp16"
echo "[export] ② 补 readout.safetensors（决策头=底座原件，训练未改动）..."
cp -f "$D3_MODEL_DIR/readout.safetensors" "$F16_DIR/readout.safetensors"

echo "[export] ③ 转 GGUF F16（10-20 分钟）..."
( cd "$F16_DIR" && "$D3_PY" "$D3_CONV_DIR/convert_hf_to_gguf.py" \
    --outfile "../$NAME-f16.gguf" --outtype f16 . 2>&1 | tail -5 )
F16="$MERGED/$NAME-f16.gguf"
[[ -f "$F16" ]] || { echo "[export] FAIL: F16 未生成"; exit 1; }
F16_G=$(du -g "$F16" | cut -f1)
(( F16_G >= 15 )) || { echo "[export] FAIL: F16 仅 ${F16_G}G（基准≈17.9G）"; exit 1; }

echo "[export] ④ 量化 Q8_0..."
"$D3_LLAMA_DIR/llama-quantize" "$F16" "$MERGED/$NAME-Q8_0.gguf" Q8_0 | tail -3
Q8="$MERGED/$NAME-Q8_0.gguf"
[[ -f "$Q8" ]] || { echo "[export] FAIL: Q8_0 未生成"; exit 1; }

( cd "$MERGED" && sha256sum "$NAME-Q8_0.gguf" | tee sha256-Q8_0.txt )
echo "[export] 完成:"
ls -lh "$F16" "$Q8"
echo "[export] 部署文件: $Q8"
