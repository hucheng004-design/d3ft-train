#!/usr/bin/env bash
# install_offline.sh — 服务器离线安装（配合 prepare_offline.sh 产物，全程不联网）
# 用法: bash install_offline.sh [offline目录]     # 默认 ./offline（即 d3-kit/offline）
# 自动: 校验清单 → python3.11+离线依赖 → 底座模型 → llama.cpp → 转换工具链 → 训练补丁 → 终检
# 路线A: 裸机安装（glibc≥2.28）
# 路线B: CentOS7/glibc<2.28 → Docker 容器内也可跑本脚本（传 -e D3_PY=python3，见手册第 5 节）
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

say() { printf '\n=== %s ===\n' "$*"; }
[[ "$(uname -m)" == "x86_64" ]] || { echo "FAIL: 仅支持 x86_64"; exit 1; }
OFF="${1:-$D3_KIT/offline}"
[[ -d "$OFF" ]] || { echo "FAIL: 找不到离线包目录 $OFF（prepare_offline.sh 产物，默认放 d3-kit/offline）"; exit 1; }

say "⓪ 校验离线包完整性"
if [[ -f "$OFF/MANIFEST.sha256" ]]; then
  ( cd "$OFF" && sha256sum -c MANIFEST.sha256 --quiet ) \
    || { echo "  FAIL: 校验失败，文件损坏/缺失，请重新传输"; exit 1; }
  echo "  校验通过"
else
  echo "  WARN: 无 MANIFEST.sha256（云端打包模式），跳过整体校验"
fi

say "① Python 3.11 + 离线依赖"
if [[ -n "${D3_PY:-}" ]] && "$D3_PY" -c "import torch" >/dev/null 2>&1; then
  PY="$D3_PY"
  echo "  $PY 已可用（Docker 镜像预装场景），跳过依赖安装"
else
  PREFIX="$HOME/miniconda3"
  if [[ -x "$PREFIX/bin/python" ]] && "$PREFIX/bin/python" -V 2>&1 | grep -q "3\.11"; then
    echo "  复用现有 $PREFIX（Python 3.11）"
  else
    PREFIX="$HOME/miniconda3-py311"
    if [[ ! -x "$PREFIX/bin/python" ]]; then
      CONDA_SH=$(find "$OFF/conda" -name 'Miniconda3-*Linux-x86_64.sh' | head -1)
      [[ -n "$CONDA_SH" ]] || { echo "  FAIL: offline/conda 里没有安装器"; exit 1; }
      bash "$CONDA_SH" -b -p "$PREFIX"
    fi
    echo "  已安装 $PREFIX（$("$PREFIX/bin/python" -V 2>&1)）"
  fi
  PY="$PREFIX/bin/python"
  echo "  离线安装依赖（--no-index，只用本地 wheels）..."
  "$PY" -m pip install -q --no-index --find-links="$OFF/pip-wheels" \
    "torch==2.8.0" "torchvision==0.23.0" "transformers==5.17.0" accelerate safetensors "huggingface_hub>=1.0" \
    "flash-linear-attention==0.5.2" einops pillow opencv-python-headless \
    bitsandbytes peft sentencepiece
  grep -q "d3-offline-py" env.sh || cat >> env.sh <<EOF

# >>> d3-offline-py（install_offline.sh 生成；离线 python 路径，删掉此段即恢复默认）
export D3_PY="$PREFIX/bin/python"
# <<<
EOF
  export D3_PY="$PY"
fi
echo "  python: $PY"

say "② 底座模型 -> $D3_MODEL_DIR"
mkdir -p "$D3_MODEL_DIR"
MODEL_OK=1
if [[ -f "$D3_MODEL_DIR/decision_config.json" ]]; then
  echo "  已存在，跳过"
elif [[ -f "$OFF/model/d3-flash/decision_config.json" ]]; then
  echo "  拷贝约 17G，稍等..."
  cp -a "$OFF/model/d3-flash/." "$D3_MODEL_DIR/"
else
  MODEL_OK=0
  echo "  WARN: 离线包/镜像不含模型（云端打包模式），请手动下载后放到: $D3_MODEL_DIR"
  echo "    有网机器: pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "              modelscope download --model vllm-sr/d3-flash --local_dir d3-flash"
  echo "    把 d3-flash 整目录传到服务器 ~/d3-work/models/d3-flash/，然后重跑本脚本（幂等）补训练补丁"
fi
if (( MODEL_OK )); then
  ls "$D3_MODEL_DIR"/model*.safetensors >/dev/null 2>&1 \
    || { echo "  FAIL: model*.safetensors 缺失，检查 $D3_MODEL_DIR"; exit 1; }
  echo "  模型文件齐全"
fi

say "③ llama.cpp -> $D3_LLAMA_DIR"
if [[ -x "$D3_LLAMA_DIR/llama-server" ]]; then
  echo "  已存在，跳过"
else
  Z=$(find "$OFF/llama" -name '*bin-ubuntu-x64.zip' | head -1)
  [[ -n "$Z" ]] || { echo "  FAIL: offline/llama 里没有发行版 zip"; exit 1; }
  TMP=$(mktemp -d)
  "$PY" -m zipfile -e "$Z" "$TMP"
  SRV=$(find "$TMP" -type f -name llama-server | head -1)
  [[ -n "$SRV" ]] || { echo "  FAIL: zip 里没找到 llama-server"; exit 1; }
  mkdir -p "$D3_LLAMA_DIR" && cp -a "$(dirname "$SRV")/." "$D3_LLAMA_DIR/"
  rm -rf "$TMP"
fi
if LD_LIBRARY_PATH="$D3_LLAMA_DIR" "$D3_LLAMA_DIR/llama-server" --version >/dev/null 2>&1; then
  echo "  llama-server 可运行"
else
  echo "  WARN: llama-server 跑不起来（多为 glibc 太老）。量化/验证改在 Docker 容器里做，"
  echo "        或用 offline/llama/llama.cpp-b11588.tar.gz 在兼容环境源码编译"
fi

say "④ 转换工具链 -> $D3_CONV_DIR"
mkdir -p "$D3_CONV_DIR"
cp -r "$D3_KIT/llamacpp-convert/." "$D3_CONV_DIR/"
echo "  OK"

say "⑤ 训练前置补丁（幂等）"
if [[ -f "$D3_MODEL_DIR/decision_config.json" ]]; then
  "$D3_PY" "$D3_KIT/patch/patch_d3_cache.py" --model-dir "$D3_MODEL_DIR" || true
else
  echo "  跳过（模型未就位；模型放好后重跑本脚本即可补上）"
fi

say "⑥ 终检（导入验证）"
PYOK=1
"$PY" - <<'PYEOF' 2>&1 || PYOK=0
import torch, transformers, peft, bitsandbytes
assert transformers.__version__ == "5.17.0"
print(f"torch {torch.__version__} | transformers {transformers.__version__} | cuda可见={torch.cuda.is_available()}")
PYEOF
if (( ! PYOK )); then
  cat <<'TIP'

  FAIL: 依赖导入失败。最常见原因: 系统 glibc < 2.28（torch≥2.8 官方 wheel 要求 glibc≥2.28）。
  CentOS 7（glibc 2.17）裸机装不上 torch —— 请走【路线B: Docker 训练】:
    1. 准备机补打包:  bash prepare_offline.sh d3-offline --docker
    2. 服务器载入:    docker load -i d3-kit/offline/d3ft-train-image.tar
    3. 进容器安装（详见手册第 5 节）:
       docker run --gpus all --rm -it \
         -v ~/d3-work:/root/d3-work -v ~/d3-kit:/root/d3-kit \
         -e D3_WORK=/root/d3-work -e D3_PY=python3 d3ft-train:latest bash
       # 容器内: bash /root/d3-kit/install_offline.sh /root/d3-kit/offline
TIP
  exit 1
fi

say "离线安装完成"
echo "  python : $PY"
echo "  模型   : $D3_MODEL_DIR"
echo "  llama  : $D3_LLAMA_DIR"
echo "  工具链 : $D3_CONV_DIR"
echo "  复查: bash check_env.sh（网络 FAIL 属离线预期，忽略）"
echo "  下一步: 按手册第 6 节准备数据 → 第 7 节训练"
