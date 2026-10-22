# d3-kit — d3-flash 决策模型训练部署工具包（可移植）

> 放到任何一台新服务器（Linux x86_64 + NVIDIA GPU）即可从零完成：装环境 → 下载模型 → 训练 → 导出 GGUF → llama.cpp 上线。
> 完整操作手册见上级目录《决策模型部署训练手册-新服务器.md》。

## 快速开始（新服务器，4 条命令）

```bash
# 0. 环境预检（只读，列出缺失清单与修复提示；退出码 1 = 有阻断项）
bash check_env.sh
# 无外网服务器：先在有网准备机跑 bash prepare_offline.sh [--docker] 打离线包，
# 传到服务器后 bash install_offline.sh 离线安装（详见手册第 5 节）

# 1. 整包上传后，初始化（装 conda env/解压 llama.cpp/下载模型 ~17G，幂等）
bash bootstrap.sh

# 2. 准备数据（L3 格式校验 + holdout 切分 + L4 转换）
bash scripts/validate_data.sh data/你的标注.jsonl

# 3. 训练（先 --dry-run 验证，再正式训练；约占 GPU 7GB，别和推理服务同卡混跑）
bash scripts/train.sh --dry-run
bash scripts/train.sh

# 4. 导出 + 部署 + 验证
bash scripts/export.sh output/<训练目录>/adapter-final v1
bash scripts/serve.sh daemon                 # 或 install-systemd 注册系统服务
python3 scripts/evaluate.py --api http://127.0.0.1:8283
```

## 目录结构

```
d3-kit/
├── env.sh                    # 环境变量集中配置（路径/端口/超参，按需改）
├── check_env.sh              # 环境预检（只读；OK/WARN/FAIL/缺失 四级输出，--pre 快速模式）
├── bootstrap.sh              # 新服务器一键初始化（幂等）
├── prepare_offline.sh        # 离线包制作（有网准备机上跑；--docker 导出训练镜像）
├── install_offline.sh        # 离线安装（无外网服务器；容器内亦可跑）
├── download_model.sh         # ModelScope 下载底座模型（断点续传+sha256）
├── runtime/llama-b11588-linux.tar.gz   # llama.cpp 预编译包（需 glibc≥2.38；低版本自动转官方发行版/源码构建）
├── llamacpp-convert/         # d3 支持版 GGUF 转换工具链（convert_hf_to_gguf.py + conversion/ + gguf-py）
├── patch/patch_d3_cache.py   # 训练前置补丁（幂等；4bit 量化加载支持）
├── l3_to_l4/                 # L3→L4 数据转换包（load_codes 已修复为读模型目录 decision_config）
├── scripts/
│   ├── validate_data.sh/.py  # 数据校验 + holdout 切分 + L4 转换
│   ├── train.sh + train_d3.py        # 4bit + LoRA 训练
│   ├── export.sh + merge_lora_to_gguf.py + export_readout.py   # 合并导出 GGUF
│   ├── serve.sh + d3-server.service  # llama.cpp 服务管理（前台/后台/systemd）
│   └── evaluate.py           # 模型评估（holdout/抗遗忘，PASS/FAIL 报告）
└── examples/sample_l3.jsonl  # 数据格式示例（noul/choice/score 各一）
```

## 硬件与系统要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA ≥16GB 显存（Q8_0 占 ~9.5G；训练占 ~7G，两者不要同卡同时跑） |
| 系统 | Linux x86_64；glibc ≥2.38 可用打包二进制，否则自动走官方发行版/源码构建 |
| 磁盘 | ≥60G 空闲（模型 17G + F16 中间产物 18G + Q8 9G + 工作区） |
| 内存 | ≥30G（训练补丁预加载需一次性载入模型） |
| 网络 | 能访问 modelscope.cn（模型下载）；GitHub 可选 |

## 数据格式（L3，一行一样本）

```json
{"state": {...业务数据...},
 "questions": {"规则名": {"type": "noul|choice|score", "instructions": "问题", "criteria": {...}}},
 "answers":   {"规则名": {"type": "...", "noul": 0.9 | "choice": "选项key" | "score": 2}}}
```
- noul 取值 0~1：≥0.75=是、≤0.25=否、其余=不确定；choice 填 criteria 的 key；score 填 0 起始级别号

## 关键协议（llama.cpp 服务调用，实测验证）

```bash
curl http://127.0.0.1:8283/v1/chat/completions -H "Authorization: Bearer $D3_API_KEY" -d '{
  "messages": [{"role":"user","content":"State:\n{...}\n\nQuestion:\n...\n\nOptions:\nA: ...\nB: ...\n\nReply with only the code of the best option."}],
  "temperature": 0, "max_tokens": 1,
  "chat_template_kwargs": {"enable_thinking": false}}'
# 响应 content 中首个字母即决策 code（A/B/C...），按 Options 顺序对应答案
```
（三段式渲染直接用 `l3_to_l4/render.py` 的 `render_body()` 生成，勿手拼）

## 注意事项

- **训练前必须停同卡推理服务**（16G 卡装不下 7G 训练 + 9.5G 推理）
- `load_codes()` 必须能读到模型目录的 `decision_config.json`（env.sh 的 `D3_MODEL_DIR` 要正确）
- 首次训练会自动生成 trust_remote_code 缓存并打 4bit 补丁（`patch/patch_d3_cache.py`，幂等）
- 版本升级重训时混入 ≥5% 旧类别样本防遗忘（v1 实测：纯新数据训练会把旧规则准确率冲到 80%）
