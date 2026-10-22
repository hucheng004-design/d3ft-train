#!/usr/bin/env bash
# prepare_offline.sh — 在【有网的 Linux x86_64 准备机】上打离线包（服务器无外网时用）
# 产物: <输出目录>/  = Miniconda(py3.11) + Python依赖wheels + 底座模型 + llama.cpp兜底 + 校验清单
#       加 --docker 额外导出训练镜像 d3ft-train-image.tar（CentOS7/glibc<2.28 服务器走 Docker 训练）
# 用法: bash prepare_offline.sh [输出目录] [--docker]      默认输出 ./d3-offline
# 要求: Linux x86_64 + python3.11（无则自动用 conda 建一个）+ 磁盘 ≥40G；--docker 需 Docker
# 说明: 模型下载支持断点续传，中断后重跑本脚本即可
set -euo pipefail

OUT="d3-offline"; DOCKER=0
for a in "$@"; do
  [[ "$a" == "--docker" ]] && DOCKER=1 || OUT="$a"
done
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || { echo "FAIL: 请在 Linux x86_64 准备机上运行"; exit 1; }
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"
DEPS=("\"torch==2.8.0\"" "torchvision==0.23.0" "transformers==5.17.0" accelerate safetensors "huggingface_hub>=1.0"
      "flash-linear-attention==0.5.2" einops pillow opencv-python-headless
      bitsandbytes peft sentencepiece)

echo "== ① 准备 python3.11（pip download 的 wheel 版本必须与服务器一致）=="
PY11=""
if command -v python3.11 >/dev/null 2>&1; then
  PY11="$(command -v python3.11)"
elif command -v conda >/dev/null 2>&1; then
  conda env list | grep -q "^d3pack " || conda create -y -n d3pack python=3.11
  PY11="$(conda run -n d3pack which python 2>/dev/null | tail -1)"
fi
[[ -n "$PY11" ]] || { echo "FAIL: 没有 python3.11 也无 conda，先装 Miniconda（https://docs.conda.io）"; exit 1; }
echo "  使用: $PY11 ($("$PY11" -V 2>&1))"

mkdir -p "$OUT"/{conda,pip-wheels,model,llama}

echo "== ② Miniconda py311 安装器（服务器离线装，base 即 Python 3.11）=="
CONDA_SH="$OUT/conda/Miniconda3-py311-Linux-x86_64.sh"
if [[ ! -s "$CONDA_SH" ]]; then
  for v in 24.11.1-0 24.9.2-0 24.7.1-0 24.5.0-0 24.3.0-0 24.1.2-0; do
    curl -fL --retry 2 -o "$CONDA_SH" \
      "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_${v}-Linux-x86_64.sh" \
      && break || rm -f "$CONDA_SH"
  done
fi
[[ -s "$CONDA_SH" ]] || { echo "FAIL: py311 安装器下载失败，去 https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/ 挑 py311 版本手动放到 $CONDA_SH"; exit 1; }
echo "  OK: $(du -h "$CONDA_SH" | cut -f1)"

echo "== ③ Python 依赖 wheels（torch+CUDA运行库全家桶，约 5~8G）=="
"$PY11" -m pip download -d "$OUT/pip-wheels" -i "$TUNA" "${DEPS[@]}"

echo "== ④ 底座模型 vllm-sr/d3-flash（约 17G，断点续传，中断重跑即可）=="
"$PY11" -m pip install -q -U modelscope -i "$TUNA"
"$(dirname "$PY11")/modelscope" download --model vllm-sr/d3-flash --local_dir "$OUT/model/d3-flash"

echo "== ⑤ llama.cpp 兜底（官方 ubuntu 发行版 + 源码包）=="
LLZIP="$OUT/llama/llama-b11588-bin-ubuntu-x64.zip"
if [[ ! -s "$LLZIP" ]]; then
  curl -fL --retry 2 -o "$LLZIP" \
    "https://github.com/ggml-org/llama.cpp/releases/download/b11588/llama-b11588-bin-ubuntu-x64.zip" \
    || curl -fL --retry 2 -o "$LLZIP" \
    "https://ghproxy.net/https://github.com/ggml-org/llama.cpp/releases/download/b11588/llama-b11588-bin-ubuntu-x64.zip" \
    || { rm -f "$LLZIP"; echo "  WARN: 发行版下载失败（不影响 Docker 路线）"; }
fi
[[ -s "$OUT/llama/llama.cpp-b11588.tar.gz" ]] || \
  curl -fL --retry 2 -o "$OUT/llama/llama.cpp-b11588.tar.gz" \
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11588.tar.gz" \
    || curl -fL --retry 2 -o "$OUT/llama/llama.cpp-b11588.tar.gz" \
    "https://ghproxy.net/https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11588.tar.gz" \
    || { rm -f "$OUT/llama/llama.cpp-b11588.tar.gz"; echo "  WARN: 源码包下载失败"; }

if (( DOCKER )); then
  echo "== ⑥ 构建并导出训练镜像 d3ft-train:latest（python3.11 + 依赖全预装，不含底座模型）=="
  cat > "$OUT/Dockerfile" <<EOF
# syntax=docker/dockerfile:1
FROM python:3.11-bookworm
RUN --mount=type=bind,source=pip-wheels,target=/tmp/wheels \
    pip install --no-index --find-links=/tmp/wheels ${DEPS[*]}
EOF
  DOCKER_BUILDKIT=1 docker build -t d3ft-train:latest "$OUT"
  docker save d3ft-train:latest | gzip > "$OUT/d3ft-train-image.tar.gz"
  docker rmi d3ft-train:latest >/dev/null 2>&1 || true
fi

echo "== ⑦ 生成校验清单 =="
( cd "$OUT" && find . -type f ! -name MANIFEST.sha256 -exec sha256sum {} \; > MANIFEST.sha256 )

echo "== 打包完成: $OUT =="
du -sh "$OUT"/* 2>/dev/null || true
cat <<TIP

下一步:
  1. 离线包放进工具包一起传到服务器:
       cp -r $OUT d3-kit/offline
       scp -r d3-kit user@服务器:~/          # 或内网 rsync / 移动硬盘
  2. 服务器上安装（全程离线）:
       glibc≥2.28（路线A 裸机）   : bash d3-kit/install_offline.sh
       CentOS7/glibc<2.28（路线B）: 见手册第 5 节，docker load 后进容器跑 install_offline.sh
TIP
