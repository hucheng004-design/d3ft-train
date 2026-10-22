#!/usr/bin/env bash
# check_env.sh — 训练环境预检（只读检查，不修改任何东西）
# 用法:
#   bash check_env.sh         全量体检: 硬件/系统/网络 + conda依赖/llama.cpp/转换工具链/底座模型
#   bash check_env.sh --pre   只查硬件/系统/网络（bootstrap 前快速把关）
# 输出标记: OK=就绪  WARN=能用但有隐患  FAIL=阻断项需先手动处理  缺失=由 bootstrap.sh 自动补齐
# 退出码: 0=无 FAIL  1=有 FAIL（先解决再跑 bootstrap）
set -uo pipefail
cd "$(dirname "$0")" && source ./env.sh

PRE=0; [[ "${1:-}" == "--pre" ]] && PRE=1
PASS_N=0; WARN_N=0; FAIL_N=0; TODO_N=0

say()  { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  [ \033[32mOK\033[0m ] %s\n' "$*";  PASS_N=$((PASS_N+1)); }
warn() { printf '  [\033[33mWARN\033[0m] %s\n' "$*";  WARN_N=$((WARN_N+1)); }
bad()  { printf '  [\033[31mFAIL\033[0m] %s\n' "$*";  FAIL_N=$((FAIL_N+1)); }
todo() { printf '  [\033[36m缺失\033[0m] %s\n' "$*";  TODO_N=$((TODO_N+1)); }

say "① 硬件（GPU/磁盘/内存/CPU）"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ROW=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1)
  if [[ -n "${GPU_ROW:-}" ]]; then
    GPU_CNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    ok "GPU x${GPU_CNT}: ${GPU_ROW}"
    DRV=$(awk -F', *' '{print $NF}' <<<"$GPU_ROW")
    DRV_MAJOR=${DRV%%.*}
    (( DRV_MAJOR >= 525 )) || warn "驱动 ${DRV} 偏老：torch cu126 建议 ≥525.60（Linux）"
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
    (( GPU_MEM_MB >= 15000 )) || warn "单卡显存 ${GPU_MEM_MB}MB < 16G：训练需 ~7G，Q8 验证需 ~9.5G"
  else
    warn "nvidia-smi 存在但无输出，驱动可能异常（手动跑 nvidia-smi 确认）"
  fi
else
  warn "无 NVIDIA GPU/驱动：训练与 GPU 验证不可用"
fi

DISK_TARGET="$D3_WORK"; [[ -d "$D3_WORK" ]] || DISK_TARGET="$HOME"
DISK_G=$(df -BG --output=avail "$DISK_TARGET" 2>/dev/null | tail -1 | tr -dc 0-9)
if [[ -n "${DISK_G:-}" ]] && (( DISK_G >= 60 )); then
  ok "磁盘可用 ${DISK_G}G（${DISK_TARGET}，要求 ≥60G）"
else
  bad "磁盘可用 ${DISK_G:-?}G（${DISK_TARGET}）< 60G，先清理空间"
fi

MEM_G=$(free -g 2>/dev/null | awk '/^Mem/{print $2}')
if [[ -n "${MEM_G:-}" ]]; then
  if   (( MEM_G >= 28 )); then ok "内存 ${MEM_G}G"
  elif (( MEM_G >= 16 )); then warn "内存 ${MEM_G}G 偏小（训练补丁预加载建议 ≥30G）"
  else bad "内存 ${MEM_G}G 过小（建议 ≥30G）"; fi
fi

NPROC=$(nproc 2>/dev/null || echo 0)
(( NPROC >= 8 )) && ok "CPU ${NPROC} 核" || warn "CPU ${NPROC} 核偏少（建议 ≥8 核）"

say "② 系统（架构/glibc/基础命令）"
ok "系统: $(uname -s) $(uname -m)，内核 $(uname -r)"
[[ "$(uname -s)" == "Linux" ]] || bad "系统 $(uname -s) 非 Linux（本 kit 仅支持 Linux x86_64）"
ARCH=$(uname -m)
[[ "$ARCH" == "x86_64" ]] || bad "架构 ${ARCH} 不支持（需 x86_64）"
GLIBC=$(ldd --version 2>/dev/null | awk 'NR==1{gsub(/\./,"",$NF); print $NF}')
if [[ -n "${GLIBC:-}" ]] && (( GLIBC >= 238 )); then
  ok "glibc ≥2.38：可直接解压 runtime/ 打包二进制"
elif [[ -n "${GLIBC:-}" ]] && (( GLIBC >= 228 )); then
  warn "glibc ${GLIBC} < 2.38：bootstrap 自动改下官方发行版（需 GitHub）或源码构建"
else
  warn "glibc ${GLIBC:-?} < 2.28（如 CentOS 7）：torch≥2.8 装不上，训练走手册第 5 节路线B（Docker）"
fi
MISS_CMD=()
for t in curl tar unzip sha256sum awk find; do
  command -v "$t" >/dev/null 2>&1 || MISS_CMD+=("$t")
done
if (( ${#MISS_CMD[@]} == 0 )); then ok "基础命令齐全（curl/tar/unzip/sha256sum/awk/find）"
else bad "缺命令: ${MISS_CMD[*]}（Ubuntu: sudo apt install ${MISS_CMD[*]}）"; fi
command -v git >/dev/null 2>&1 || warn "无 git：仅 llama.cpp 源码构建兜底时需要"

say "③ 网络（下载源可达性）"
net() { curl -fsI --max-time 8 -o /dev/null "$1" 2>/dev/null; }
if net https://modelscope.cn; then ok "modelscope.cn 可达（底座模型下载源）"
else bad "modelscope.cn 不可达：无法在线下载模型（无外网？走离线包：准备机跑 prepare_offline.sh，见手册第 5 节）"; fi
if net https://pypi.tuna.tsinghua.edu.cn/simple; then ok "清华 pip 源可达（conda/pip 依赖安装源）"
else bad "清华 pip 源不可达：依赖无法在线安装（无外网？走离线包，见手册第 5 节）"; fi
if net https://github.com; then ok "github.com 可达（llama.cpp 兜底下载源）"
elif (( GLIBC >= 238 )); then warn "github.com 不可达：仅 glibc<2.38 兜底时才需要，当前不影响"
else warn "github.com 不可达且 glibc<2.38：llama.cpp 只能源码构建或手动就位"; fi

if (( PRE )); then
  say "体检汇总（--pre 快速模式）"
  printf '  OK:%d  WARN:%d  FAIL:%d\n' "$PASS_N" "$WARN_N" "$FAIL_N"
  if (( FAIL_N )); then echo "  有阻断项：先按 FAIL 提示处理"; exit 1; fi
  echo "  硬件/系统/网络无阻断，下一步: bash bootstrap.sh（自动补齐其余依赖与产物）"
  exit 0
fi

say "④ Python 环境（conda env: $D3_CONDA_ENV）"
if [[ -x "$HOME/miniconda3/bin/conda" ]] || command -v conda >/dev/null 2>&1; then
  ok "conda 已安装"
  if conda env list 2>/dev/null | grep -q "^${D3_CONDA_ENV} "; then
    ok "conda env ${D3_CONDA_ENV} 已创建"
  else
    todo "conda env ${D3_CONDA_ENV} 未创建 → bash bootstrap.sh 自动创建"
  fi
else
  todo "conda 未安装 → bash bootstrap.sh 自动装 miniconda（清华源）"
fi
if [[ -x "$D3_PY" ]]; then
  PYOUT=$("$D3_PY" - <<'PY' 2>&1
import importlib
mods = ["torch", "transformers", "peft", "bitsandbytes", "accelerate",
        "flash_linear_attention", "safetensors", "einops"]
miss = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"    {m}: {getattr(mod, '__version__', '?')}")
    except Exception:
        miss.append(m)
warns = []
try:
    import torch, transformers
    if transformers.__version__ != "5.17.0":
        warns.append(f"transformers {transformers.__version__}（要求 5.17.0）")
    t = torch.__version__.split("+")[0].split(".")
    if tuple(int(x) for x in t[:2]) < (2, 8):
        warns.append(f"torch {torch.__version__}（要求 >=2.8）")
    if not torch.cuda.is_available():
        warns.append("torch 检测不到 CUDA（训练会极慢/失败，检查驱动与 torch 版本）")
except Exception:
    pass
print("D3WARN:" + "|".join(warns))
print("D3MISS:" + ",".join(miss))
PY
)
  grep -v '^D3' <<<"$PYOUT" || true
  W=$(grep '^D3WARN:' <<<"$PYOUT" | cut -d: -f2-)
  M=$(grep '^D3MISS:' <<<"$PYOUT" | cut -d: -f2-)
  if [[ -z "$M" ]]; then ok "Python 依赖齐全"
  else todo "缺 Python 依赖: ${M} → bash bootstrap.sh 自动补装"; fi
  if [[ -n "$W" ]]; then
    while IFS= read -r line; do [[ -n "$line" ]] && warn "$line"; done <<<"${W//|/$'\n'}"
  fi
else
  todo "训练 Python（$D3_PY）不存在 → bash bootstrap.sh 自动建环境"
fi

say "⑤ llama.cpp 运行时（量化/验证用，$D3_LLAMA_DIR）"
if [[ -x "$D3_LLAMA_DIR/llama-server" && -x "$D3_LLAMA_DIR/llama-quantize" ]]; then
  if LD_LIBRARY_PATH="$D3_LLAMA_DIR" "$D3_LLAMA_DIR/llama-server" --version >/dev/null 2>&1; then
    ok "llama.cpp 就位且可运行"
  else
    bad "llama.cpp 存在但无法运行（多为 glibc/CUDA 问题）→ 删 $D3_LLAMA_DIR 后重跑 bootstrap.sh"
  fi
else
  todo "llama.cpp 未就位（缺 llama-server/llama-quantize）→ bash bootstrap.sh 自动解压/下载"
fi

say "⑥ GGUF 转换工具链（$D3_CONV_DIR）"
if [[ -f "$D3_CONV_DIR/convert_hf_to_gguf.py" && -f "$D3_CONV_DIR/conversion/vllm_d3.py" ]]; then
  ok "转换工具链就位（convert_hf_to_gguf.py + conversion/vllm_d3.py）"
else
  todo "转换工具链未就位 → bash bootstrap.sh 自动拷贝"
fi

say "⑦ 底座模型（$D3_MODEL_DIR）"
shopt -s nullglob
SHARDS=("$D3_MODEL_DIR"/model*.safetensors)
shopt -u nullglob
if (( ${#SHARDS[@]} > 0 )) && [[ -f "$D3_MODEL_DIR/decision_config.json" && -f "$D3_MODEL_DIR/readout.safetensors" ]]; then
  MODEL_MB=$(du -sm "$D3_MODEL_DIR" 2>/dev/null | cut -f1)
  ok "底座模型就位（权重 ${#SHARDS[@]} 个分片，约 $(( ${MODEL_MB:-0} / 1024 ))G）"
else
  todo "底座模型缺失/不完整（需 model*.safetensors + decision_config.json + readout.safetensors）→ bash bootstrap.sh 自动下载（约 17G）"
fi

say "体检汇总"
printf '  OK:%d  WARN:%d  FAIL:%d  缺失:%d\n' "$PASS_N" "$WARN_N" "$FAIL_N" "$TODO_N"
if (( FAIL_N )); then
  echo "  存在阻断项（FAIL）：先按上方提示手动处理，再跑 bootstrap.sh"
  exit 1
fi
if (( TODO_N )); then
  echo "  缺失项补齐方式:"
  echo "    在线服务器     → bash bootstrap.sh（幂等，自动补齐，已就绪项自动跳过）"
  echo "    无外网服务器   → 准备机打离线包 + bash install_offline.sh（手册第 5 节）"
else
  echo "  环境全部就绪，可直接训练/导出"
fi
exit 0
