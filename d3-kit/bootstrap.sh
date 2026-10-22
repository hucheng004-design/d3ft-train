#!/usr/bin/env bash
# bootstrap.sh — 新服务器一键初始化
# 内容: 环境检查 → miniconda(可选) → conda env + pip 依赖 → llama.cpp 运行时 → 转换工具链 → 模型下载
# 用法: bash bootstrap.sh            （交互缺省安全；重复执行幂等）
set -euo pipefail
cd "$(dirname "$0")" && source ./env.sh

say() { echo -e "\n=== $* ==="; }

say "① 环境检查（详细缺失清单: bash check_env.sh，只读预检）"
fail=0
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "  WARN: 无 NVIDIA GPU/驱动，训练与 GPU 推理不可用（仅能 CPU 慢速推理）"; }
GLIBC=$(ldd --version | awk 'NR==1{gsub(/\./,"",$NF); print $NF}')
echo "  glibc: $(ldd --version | head -1)"
DISK_G=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
(( DISK_G >= 60 )) || { echo "  FAIL: 家目录剩余 ${DISK_G}G < 60G"; fail=1; }
MEM_G=$(free -g | awk '/^Mem/{print $2}')
(( MEM_G >= 24 )) || echo "  WARN: 内存 ${MEM_G}G 偏小（训练/补丁预加载建议 ≥30G）"
(( fail )) && exit 1
echo "  磁盘可用 ${DISK_G}G / 内存 ${MEM_G}G"

say "② miniconda（无 conda 时自动安装到 ~/miniconda3，清华源）"
if ! command -v conda >/dev/null && [[ ! -x "$HOME/miniconda3/bin/conda" ]]; then
  curl -fL --retry 3 -o /tmp/miniconda.sh \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3" && rm /tmp/miniconda.sh
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "  conda: $(conda --version)"

say "③ conda env $D3_CONDA_ENV + 依赖"
if ! conda env list | grep -q "^$D3_CONDA_ENV "; then
  conda create -y -n "$D3_CONDA_ENV" python=3.11
fi
PIP="$HOME/miniconda3/envs/$D3_CONDA_ENV/bin/pip"
"$PIP" install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "torch>=2.8" "transformers==5.17.0" accelerate safetensors "huggingface_hub>=1.0" \
  "flash-linear-attention==0.5.2" einops pillow torchvision opencv-python-headless \
  bitsandbytes peft sentencepiece
"$PIP" install -q -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps "transformers==5.17.0"
echo "  依赖安装完成: $("$HOME/miniconda3/envs/$D3_CONDA_ENV/bin/python" -c "import torch,transformers,peft,bitsandbytes;print('torch',torch.__version__,'| tf',transformers.__version__)")"

say "④ llama.cpp 运行时 -> $D3_LLAMA_DIR"
mkdir -p "$D3_WORK"
if [[ -x "$D3_LLAMA_DIR/llama-server" ]]; then
  echo "  已存在，跳过"
elif (( GLIBC >= 238 )) && [[ -f "$D3_KIT/runtime/llama-b11588-linux.tar.gz" ]]; then
  tar xzf "$D3_KIT/runtime/llama-b11588-linux.tar.gz" -C "$D3_WORK"
  echo "  已解压打包二进制（glibc $GLIBC 兼容）"
else
  echo "  打包二进制需 glibc≥2.38（本机 $GLIBC），改用官方 GitHub 发行版..."
  for url in \
    "https://github.com/ggml-org/llama.cpp/releases/download/b11588/llama-b11588-bin-ubuntu-x64.zip" \
    "https://ghproxy.net/https://github.com/ggml-org/llama.cpp/releases/download/b11588/llama-b11588-bin-ubuntu-x64.zip"; do
    curl -fL --retry 2 -o /tmp/llama.zip "$url" && break || true
  done
  if [[ -s /tmp/llama.zip ]]; then
    unzip -q -o /tmp/llama.zip -d "$D3_WORK" && rm /tmp/llama.zip
  else
    echo "  FAIL: 发行版下载失败。手动方案:"
    echo "    git clone --depth 1 -b b11588 https://github.com/ggml-org/llama.cpp $D3_WORK/llama.cpp && \\"
    echo "    cd $D3_WORK/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
    echo "    并把 build/bin/* 拷到 $D3_LLAMA_DIR/"
    exit 1
  fi
fi
LD_LIBRARY_PATH="$D3_LLAMA_DIR" "$D3_LLAMA_DIR/llama-server" --version 2>&1 | head -1 \
  || { echo "  FAIL: llama-server 无法运行（见上方报错）"; exit 1; }

say "⑤ GGUF 转换工具链 -> $D3_CONV_DIR"
mkdir -p "$D3_CONV_DIR"
cp -r "$D3_KIT/llamacpp-convert/." "$D3_CONV_DIR/"
find "$D3_CONV_DIR" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
echo "  convert_hf_to_gguf.py 就位"

say "⑥ 训练前置补丁（幂等）"
"$D3_PY" "$D3_KIT/patch/patch_d3_cache.py" --model-dir "$D3_MODEL_DIR" || true

say "⑦ 下载底座模型 -> $D3_MODEL_DIR"
bash "$D3_KIT/download_model.sh" "$D3_MODEL_DIR"

say "bootstrap 完成"
cat <<TIP
目录结构:
  $D3_WORK/
  ├── models/d3-flash/      底座模型（~17G）
  ├── llama-b11588/         llama.cpp 运行时
  └── llamacpp-convert/     GGUF 转换工具链

下一步:
  1. 准备数据:   cp 你的.jsonl $D3_WORK/data/ && bash $D3_KIT/scripts/validate_data.sh data/xxx.jsonl
  2. 训练:       bash $D3_KIT/scripts/train.sh
  3. 导出+部署:  bash $D3_KIT/scripts/export.sh && bash $D3_KIT/scripts/serve.sh start
完整说明见《决策模型部署训练手册-新服务器.md》
TIP
