#!/usr/bin/env bash
# download_model.sh — 从 ModelScope 下载 d3-flash 底座模型（断点续传 + sha256 校验）
# 用法: bash download_model.sh [目标目录]   （缺省 $D3_MODEL_DIR）
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

DEST="${1:-$D3_MODEL_DIR}"
BASE_URL="https://www.modelscope.cn/models/vllm-sr/d3-flash/resolve/master"

FILES=(
  config.json configuration.json decision_config.json chat_template.jinja
  tokenizer.json tokenizer_config.json vocab.json merges.txt
  preprocessor_config.json video_preprocessor_config.json
  model.safetensors.index.json
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
  readout.safetensors
  modeling_d3.py d3_engine.py d3_fast.py d3_format.py d3_kernels.py
  d3_server.py pipeline_d3.py requirements.txt LICENSE NOTICE
)

mkdir -p "$DEST"
echo "[model] 目标目录: $DEST"

# 生成官方 sha256 清单（用于大文件校验；失败不影响下载）
SUMFILE="$DEST/.sha256sums"
if command -v python3 >/dev/null 2>&1; then
  curl -sfL "https://www.modelscope.cn/api/v1/models/vllm-sr/d3-flash/repo/files?Revision=master&Root=" \
    | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    for f in d["Data"]["Files"]:
        if f.get("Sha256"):
            print(f["Sha256"], f["Path"])
except Exception:
    pass' > "$SUMFILE" || true
  [[ -s "$SUMFILE" ]] && echo "[model] sha256 清单: $(wc -l < "$SUMFILE") 项" || rm -f "$SUMFILE"
fi

check_sha() {  # check_sha <file>
  [[ -s "$SUMFILE" ]] || return 0
  local want got
  want=$(awk -v p="$(basename "$1")" '$2==p{print $1}' "$SUMFILE")
  [[ -n "$want" ]] || return 0
  got=$(sha256sum "$1" | awk '{print $1}')
  if [[ "$want" != "$got" ]]; then
    echo "[model] sha256 不一致: $(basename "$1")"; return 1
  fi
  echo "[model] sha256 OK: $(basename "$1")"
}

fail=0
for f in "${FILES[@]}"; do
  out="$DEST/$f"
  if [[ -s "$out" && "$f" == *.safetensors ]]; then
    check_sha "$out" || { rm -f "$out"; fail=1; }
    continue
  fi
  echo "[model] 下载: $f"
  curl -fL -C - --retry 5 --retry-delay 3 -o "$out" "$BASE_URL/$f" || { echo "[model] FAIL: $f"; fail=1; continue; }
done

# 全量大文件复验
for f in "$DEST"/*.safetensors; do
  [[ -e "$f" ]] || continue
  check_sha "$f" || fail=1
done

# 关键文件齐套检查
for f in config.json decision_config.json readout.safetensors \
         model-00001-of-00004.safetensors model-00004-of-00004.safetensors; do
  [[ -s "$DEST/$f" ]] || { echo "[model] 缺关键文件: $f"; fail=1; }
done
(( fail )) && { echo "[model] 未完成，重跑本脚本可断点续传"; exit 1; }

echo "[model] 完成: $(du -sh "$DEST" | cut -f1) @ $DEST"
