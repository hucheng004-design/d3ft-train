# l3_to_l4 — 决策模型训练数据转换工具

**版本**: 0.2.0 (B 精简版)
**目的**: 将 L3（原始标注）转换为 L4（训练就绪），与 `train_d3.py` 配套使用

## 1. 什么是 L3 / L4

| 层级 | 定义 | 文件 |
|---|---|---|
| **L3** | 你的原始标注数据（d3 schema：state + questions + answers）| `l3_data/*.jsonl` |
| **L4** | 渲染后的训练数据（B 精简版：4 顶层字段 + 5 字段 meta）| `l4_data/*.l4.jsonl` |

**L4 = L3 经过两步转换**：
1. **expand** — 一行 L3（多 question）拆成 N 行 L3.5（单 question）
2. **render** — 每个 L3.5 拼成 d3 chat prompt（`render_chat()`）

## 2. 安装 / 路径

**Windows 端**：`C:\文档\邵震洲\训练\l3_to_l4\`
**WSL 端**：`~/d3-finetune/l3_to_l4\`（同步）

```python
import sys
sys.path.insert(0, r"C:\Users\shaoz\l3_to_l4")  # Windows
# 或
sys.path.insert(0, "/home/d3ft/d3-finetune")  # WSL
```

## 3. CLI 用法

```bash
# 单文件转换
python -m l3_to_l4 preprocess l3_data/d3_train.jsonl l4_data/v1_train.l4.jsonl

# 批量转换
python -m l3_to_l4 batch l3_data/ l4_data/ --glob "*.jsonl"

# 校验
python -m l3_to_l4 validate l4_data/v1_train.l4.jsonl

# 统计
python -m l3_to_l4 stats l4_data/v1_train.l4.jsonl
```

## 4. Python API

```python
from l3_to_l4.preprocessor import process_file, load_codes
from l3_to_l4.validator import validate_l4_file

codes = load_codes()  # 从 base/decision_config.json 加载 255 个 codes
n = process_file("l3_data/in.jsonl", "l4_data/out.jsonl", codes)
errors = validate_l4_file("l4_data/out.jsonl", codes)
```

## 5. L4 schema (B 精简版)

```json
{
  "l3_source": "d3_train.jsonl:42",      // 来源追踪（L3↔L4 关联）
  "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n\n\n",
  "target_token_idx": 5,                // codes 列表位置（0-254）
  "meta": {
    "code_text": "F",                    // 人类可读 debug
    "question_name": "...",              // 来自 L3 question key
    "source_qkey": "...",                // 冗余备份
    "llamacpp_version": "b11588",
    "created": "2026-10-17T20:00:00+08:00"
  }
}
```

**4 顶层字段 + 5 meta 字段**。训练只读 `prompt` + `target_token_idx`。

详细规范见 `../L4-schema-详细规范.md`。

## 6. 包结构

```
l3_to_l4/
├── __init__.py           # 导出所有公开 API
├── __main__.py           # CLI 入口
├── expand.py             # expand_dataset()：L3 → L3.5
├── render.py             # render_chat()：L3.5 → 完整 prompt
├── target.py             # compute_target_idx()：answer → codes 位置
├── validator.py         # L3 / L4 schema 校验
├── preprocessor.py       # 主入口（process_file / batch_process / load_codes）
├── tests/
│   ├── __init__.py
│   └── test_e2e.py       # 端到端测试（4 个用例）
└── README.md             # 本文档
```

## 7. 与 train_d3.py 配套使用

```bash
# 1. 转换
python -m l3_to_l4 preprocess l3_data/d3_train.jsonl l4_data/v1_train.l4.jsonl

# 2. 校验
python -m l3_to_l4 validate l4_data/v1_train.l4.jsonl

# 3. 训练（用 L4，train_d3.py 不再渲染 prompt）
python scripts/train_d3.py --data l4_data/v1_train.l4.jsonl --max_steps 1000
```

## 8. 端到端测试

```bash
cd l3_to_l4
python -m tests.test_e2e
```

4 个测试用例：
1. L3 → L4 preprocess 基本流程
2. 多 question 展开
3. score 类型 target 计算
4. v1 训练数据 roundtrip（用 d3_train.jsonl）

## 9. 关键技术点

### 9.1 noul "反直觉"映射

`option_order([1, 0])` 与默认 criteria（`{false: "No / false", true: "Yes / true"}`）配合：

- 渲染时按 `[1, 0]` 排序：
  - `A: "Yes / true"`（order[0]=1，渲染第一项）
  - `B: "No / false"`（order[1]=0）
- L3 `noul: 0.85`（"是"）→ local=1 → ans_pos=order[1]=0 → codes[0]="A" → **A 字母**

**校验规则**：渲染出的 prompt 中 "A" 必须是 "Yes / true"，否则转换有 bug。

### 9.2 target_token_idx 是 codes 列表索引

**不是**真实 vocab token id。是 d3 `codes` 列表（255 个字母 A..JT）中的**位置索引**。

训练时 Collator 用 `readout[target_token_idx]` 算 logits（第 0..254 个），不是直接用真实 vocab id。

### 9.3 d3 决策模型特殊性

- d3 不是 chat 模型，是**读出模型**：从 255 个预定义 codes 中读出 1 个
- 每个 question 独立 forward（5 question = 5 个 forward）
- 输出 token 数 = 0（不生成文本，直接读出 code）

## 10. 离线运行

l3_to_l4 是**纯 Python 包**，无 GPU/llama.cpp 依赖。可以在任何 Python 3.11+ 环境跑：

```bash
# 转换 300 条 v1 数据：0.07 秒
python -m l3_to_l4 preprocess d3_train.jsonl v1.l4.jsonl
# [preprocess] read 300 L3 lines from d3_train.jsonl
# [preprocess] write 300 L4 lines to v1.l4.jsonl
# [preprocess] time: 0.07s
```

## 11. 后续扩展

- **多模态 L3**：state 含 image/video → L4 加 `<|vision_start|>` tokens
- **chat format L3**：用户用 messages 格式 → L4 转 d3 schema
- **L3 augmentation**：同 state 多 question 变体 → 训练数据增强
- **流式转换**：大文件 L3 → L4 不一次性 load
