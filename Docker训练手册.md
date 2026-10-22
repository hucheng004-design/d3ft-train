# Docker 训练手册（d3ft-train · 无外网 GPU 服务器版）

> 场景：**CentOS 7（glibc 2.17）+ 无外网 + NVIDIA GPU** 服务器上训练 d3-flash 决策模型。
> 思路：训练环境全部装进 Docker 镜像（绕开系统 glibc 限制），镜像由 GitHub Actions 云端打包；
> **底座模型、数据、训练产物都在宿主机，`-v` 挂载进容器**——容器删了数据也不丢。

---

## 0. 架构与前提

三类东西分清楚：

| 内容 | 放哪 | 说明 |
|---|---|---|
| 环境（python3.11 + torch 全家桶 + 工具） | Docker 镜像 `d3ft-train` | 只读，云端打包，不含模型 |
| 模型 + 数据 + 训练产物 | 宿主机 `~/d3-work` | `-v` 挂载进容器，持久保存 |
| 脚本工具包 d3-kit | 宿主机 `~/d3-kit` | `-v` 挂载进容器 |

服务器前提（现有推理环境已满足，缺什么补什么）：

| 项 | 要求 | 验证 |
|---|---|---|
| NVIDIA 驱动 | ≥525 | 宿主机 `nvidia-smi` 正常 |
| Docker | 任意较新版本 | `docker version` |
| nvidia-container-toolkit | 已装（推理容器在用即有） | `docker run --rm --gpus all <任意CUDA镜像> nvidia-smi` |
| 磁盘 | ≥60G 空闲（镜像 ~5G + 模型 17G + 中间产物 ~27G + 工作区） | `df -h /root` |
| 内存 | ≥30G | `free -g` |

---

## 1. 云端打包（GitHub Actions，一次性）

1. GitHub 建仓库 `d3ft-train`（Private 也可以，Actions 对私有仓库每月有免费额度，够用）
2. 把本目录（含 `.github/workflows/build-offline.yml`）和 `d3-kit/` 一起推送：

```bash
cd 决策模型/d3ft-train
git init && git add . ../d3-kit && git commit -m "init"
git remote add origin git@github.com:<你的用户名>/d3ft-train.git
git push -u origin main        # 注意把 ../d3-kit 放进仓库（如 d3-kit/ 子目录）
```

3. GitHub 页面 → **Actions** → **build-offline** → **Run workflow**
   - `include_wheels` 不用勾（那是裸机路线A才需要的）
4. 等 15~30 分钟，到仓库 **Releases** 下载资产：

| 资产 | 用途 | 是否必下 |
|---|---|---|
| `d3ft-train-image.tar.gz.part-00/01/...` | 训练镜像分卷 | **必下** |
| `d3ft-train-image.tar.gz.sha256` | 镜像整体校验 | **必下** |
| `llama-b11588-bin-ubuntu-x64.zip` | llama.cpp（量化/验证用） | **必下** |
| `MANIFEST.sha256` | 整包清单 | 可选 |
| `llama.cpp-b11588.tar.gz` | 源码兜底 | 可不下 |
| `Miniconda3-py311-*.sh` | 裸机路线A才用 | 可不下 |

---

## 2. 传到服务器并还原镜像

```bash
# ① 下载机（有网的任意机器）合并分卷并校验：
cat d3ft-train-image.tar.gz.part-* > d3ft-train-image.tar.gz
sha256sum -c d3ft-train-image.tar.gz.sha256      # 显示 OK 再传

# ② 传到服务器（~4.5G，scp / rsync / 移动硬盘均可）
scp d3ft-train-image.tar.gz root@服务器:~/

# ③ 服务器上载入镜像
docker load -i ~/d3ft-train-image.tar.gz
docker images | grep d3ft-train                  # 确认出现 d3ft-train:latest
rm ~/d3ft-train-image.tar.gz                     # 载入后可删，省 4.5G

# ④ 宿主机目录初始化 + llama 兜底就位
mkdir -p ~/d3-work/models ~/d3-work/data ~/d3-kit/offline/llama
# 把 Release 下载的 llama zip 放进去（容器内安装脚本会解压它）：
# ~/d3-kit/offline/llama/llama-b11588-bin-ubuntu-x64.zip
```

---

## 3. 底座模型手动下载与摆放

```bash
# ① 任意有网机器（Windows/Linux 均可，装个 python 即可）：
pip install -U modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
modelscope download --model vllm-sr/d3-flash --local_dir d3-flash   # 约 17G，断点续传，中断重跑

# ② 整目录传到服务器（建议先打包再传，文件数少速度快）：
tar czf d3-flash.tar.gz d3-flash
scp d3-flash.tar.gz root@服务器:~/

# ③ 服务器上解压到固定位置（目录名必须是 d3-flash）：
cd ~/d3-work/models && tar xzf ~/d3-flash.tar.gz && rm ~/d3-flash.tar.gz

# ④ 核对三样必需文件：
ls ~/d3-work/models/d3-flash/
#   model*.safetensors（若干分片，共~17G）
#   decision_config.json
#   readout.safetensors
```

> 想放别的位置？容器启动时加 `-e D3_MODEL_DIR=/root/d3-work/models/你的目录名`。

---

## 4. 启动训练容器（docker run）

### 4.1 标准命令

```bash
docker run --gpus all --rm -it --shm-size 8g \
  -v ~/d3-work:/root/d3-work \
  -v ~/d3-kit:/root/d3-kit \
  -e D3_WORK=/root/d3-work \
  -e D3_PY=python3 \
  d3ft-train:latest bash
```

| 参数 | 作用 |
|---|---|
| `--gpus all` | GPU 透传进容器（靠 nvidia-container-toolkit） |
| `--rm -it` | 退出即删容器 + 交互式 bash |
| `--shm-size 8g` | 共享内存调大，防偶发 shared memory 报错 |
| `-v ~/d3-work:...` | 模型/数据/产物挂载（**核心，不挂载数据会随容器删除**） |
| `-v ~/d3-kit:...` | 脚本工具包挂载 |
| `-e D3_PY=python3` | 用镜像自带的 python（依赖已预装，不需要 conda） |
| `-e D3_WORK=...` | 容器内工作根目录 |

> 双卡服务器（如 2×A40）：0 卡跑着推理服务时，训练容器加 `-e CUDA_VISIBLE_DEVICES=1` 用 1 卡，互不打架。

### 4.2 容器内首次初始化（每次 build 新镜像后跑一次即可）

```bash
# ---- 容器内 ----
bash /root/d3-kit/install_offline.sh /root/d3-kit/offline
#   自动：依赖自检(镜像已装,跳过) → 解压 llama.cpp → 转换工具链 → 训练补丁 → 导入终检
#   模型未下载时会打印 WARN 提示，下好放 ~/d3-work/models/d3-flash 后重跑一次即可

bash /root/d3-kit/check_env.sh
#   网络 FAIL 属离线预期，忽略；GPU/磁盘/内存/模型/llama.cpp 应全绿

# 快速 GPU 自检：
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4.3 每次进容器

前台交互用 4.1 的命令直接进。**训练这类长任务推荐后台常驻容器**：

```bash
# 宿主机：起常驻容器（--name 固定，方便 exec）
docker run --gpus all -d --name d3train --shm-size 8g \
  -v ~/d3-work:/root/d3-work -v ~/d3-kit:/root/d3-kit \
  -e D3_WORK=/root/d3-work -e D3_PY=python3 d3ft-train:latest sleep infinity

docker exec -it d3train bash        # 随时进出，容器不删
# 服务器重启后：docker start d3train && docker exec -it d3train bash
```

> 嫌命令长，在宿主机 `~/.bashrc` 加别名：
> `alias d3train='docker exec -it d3train bash'`

---

## 5. 容器内训练全流程

```bash
# ---- 以下全部在容器内执行（/root/d3-work = 宿主机 ~/d3-work）----

# ⓪ 训练前：确认同卡推理服务已停或已换卡（训练 ~7G + Q8 推理 ~9.5G，16G 卡装不下）
#    宿主机: docker ps 找到推理容器 → docker stop <容器名>；或训练容器用 CUDA_VISIBLE_DEVICES 换卡

# ① 准备数据：L3 标注 jsonl（≥300 条 + 混入 ≥5% 旧类别样本防遗忘）
#    宿主机放到 ~/d3-work/data/你的标注.jsonl，容器内直接可见
cd /root/d3-work
bash /root/d3-kit/scripts/validate_data.sh data/你的标注.jsonl
#    自动：schema 校验 → 统计 → train/holdout 切分(5%) → L4 渲染
#    产出: /root/d3-work/l4_data/你的标注.train.l4.jsonl

# ② dry-run：5 步验证数据格式（失败即停，别浪费训练时间）
bash /root/d3-kit/scripts/train.sh --dry-run

# ③ 正式训练（后台 + 断线安全）：
cd /root/d3-kit
VER=v1-$(date +%m%d) nohup bash scripts/train.sh > /root/d3-work/train-$VER.out 2>&1 &
tail -f /root/d3-work/output/$VER/train.log       # 实时日志（宿主机 tail 也能看，见下）
#    收敛标准: loss < 0.05 且 acc ≈ 1.00（单规则小数据通常到 0.01 量级）
#    参考基准: 300 条 / 1000 步 / A40 约 40 分钟 / 显存 ~7G

# ④ 合并导出 GGUF（LoRA 合并 → F16 → Q8_0 量化，一条命令四步自动）
bash /root/d3-kit/scripts/export.sh /root/d3-work/output/$VER/adapter-final $VER
#    产物: /root/d3-work/output/merged-$VER/d3-$VER-Q8_0.gguf（约 9.5G）← 交付部署用这个

# ⑤ 验证（交付闸门，PASS 才上线）
python3 /root/d3-kit/scripts/evaluate.py --gguf /root/d3-work/output/merged-$VER/d3-$VER-Q8_0.gguf
#    带抗遗忘集 + 快速冒烟:
python3 /root/d3-kit/scripts/evaluate.py --gguf ... --anti /root/d3-work/data/旧类别样本.jsonl --limit 20
#    新规则 holdout 准确率 ≥90% 且 旧类别 ≥90% 且 非法输出 0 → PASS（退出码 0）
```

**产物在宿主机的位置**（容器删了也在）：

```
~/d3-work/
├── output/<版本>/adapter-final        LoRA 权重（留档，可重导出）
├── output/merged-<版本>/d3-<版本>-Q8_0.gguf   交付部署的量化模型
├── output/eval-*/report.txt           验证报告
└── data/ l4_data/                     数据与切分
```

验证 PASS 后，把 Q8 GGUF 交给现有 Docker 推理部署流程上线（不在本手册范围）。

---

## 6. 常用操作速查

| 操作 | 命令 |
|---|---|
| 进常驻容器 | `docker exec -it d3train bash` |
| 起一次性容器 | 见 4.1 标准命令 |
| 宿主机直接看训练日志 | `tail -f ~/d3-work/output/<版本>/train.log`（挂载目录，无需进容器） |
| 看显存 | `nvidia-smi`（宿主机/容器内都行） |
| 停同卡推理服务 | 宿主机 `docker ps` → `docker stop <推理容器名>` |
| 训练换卡 | 容器加 `-e CUDA_VISIBLE_DEVICES=1` |
| 拷产物到本机 | 已在宿主机 `~/d3-work`，直接 scp |
| 数据/容器删除关系 | 容器随便删，`~/d3-work`、`~/d3-kit` 在宿主机不受影响 |
| 升级训练环境 | 重跑 GitHub workflow → 下载 → `docker load` 新镜像 |
| 清理磁盘 | `docker system df`；旧产物 `rm -rf ~/d3-work/output/merged-旧版本` |

---

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `docker: Error response... --gpus all` | nvidia-container-toolkit 未装/未生效：安装后 `systemctl restart docker`；先用现有推理容器验证 GPU 可用性 |
| 容器内 `torch.cuda.is_available()` False | 宿主机先 `nvidia-smi` 确认驱动；驱动 <525 需升级 |
| check_env 网络 FAIL | 离线预期，忽略即可 |
| `load_codes WARN 合成 codes` | 模型目录不对：确认 `~/d3-work/models/d3-flash/decision_config.json` 存在，或启动时 `-e D3_MODEL_DIR=...` |
| 训练 OOM | 停同卡推理容器或换卡；仍不够减小 `D3_GRAD_ACCUM`（如 `-e D3_GRAD_ACCUM=4`） |
| 决策 code 总是同一个 | 训练数据标签失衡（全 0/全 1），补另一类样本 |
| dry-run KeyError/ValueError | 数据字段不全，按 validate_data 报错行修数据 |
| 旧规则准确率掉 | 没混旧类别样本：每次重训混入 ≥5% 旧样本，训练前先跑一次旧样本评估留档 |
| evaluate 端口被占 | 启动容器加 `-e D3_EVAL_PORT=8292` |
| 磁盘不足 | `docker system prune`（注意会删未使用镜像）+ 清理 `~/d3-work/output` 旧版本 |

---

*版本：2026-10-21 v1.0 · 配套 d3-kit · 完整背景见《决策模型部署训练手册-新服务器版.md》第 5 节*
