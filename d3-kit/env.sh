# env.sh — d3-kit 环境配置（所有脚本 source 本文件；按需修改后生效）

# 工作根目录（数据/产物/运行时都放这里）
export D3_WORK="${D3_WORK:-$HOME/d3-work}"

# 底座模型目录（download_model.sh 的下载目标）
export D3_MODEL_DIR="${D3_MODEL_DIR:-$D3_WORK/models/d3-flash}"

# llama.cpp 运行时（llama-server / llama-quantize 所在目录）
export D3_LLAMA_DIR="${D3_LLAMA_DIR:-$D3_WORK/llama-b11588}"

# GGUF 转换工具链（llamacpp-convert，含 d3 支持的 convert_hf_to_gguf.py）
export D3_CONV_DIR="${D3_CONV_DIR:-$D3_WORK/llamacpp-convert}"

# conda 环境
export D3_CONDA_ENV="${D3_CONDA_ENV:-d3ft}"
# python 解析顺序：已激活环境 > conda env > 系统 python
export D3_PY="${D3_PY:-$HOME/miniconda3/envs/$D3_CONDA_ENV/bin/python}"

# 端口与服务
export D3_SERVE_PORT="${D3_SERVE_PORT:-8283}"     # 线上服务端口
export D3_EVAL_PORT="${D3_EVAL_PORT:-8291}"       # 验证用临时端口
export D3_API_KEY="${D3_API_KEY:-sk-change-me}"   # llama-server --api-key

# 训练默认超参
export D3_MAX_STEPS="${D3_MAX_STEPS:-1000}"
export D3_GRAD_ACCUM="${D3_GRAD_ACCUM:-8}"
export D3_LORA_R="${D3_LORA_R:-8}"
export D3_LR="${D3_LR:-1e-4}"
export D3_HOLDOUT_RATIO="${D3_HOLDOUT_RATIO:-0.05}"

# 验收阈值
export D3_PASS_ACC="${D3_PASS_ACC:-0.90}"

# kit 根目录（本文件所在目录）
export D3_KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 便捷别名
d3_python() { "$D3_PY" "$@"; }
