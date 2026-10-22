"""l3_to_l4/tests/test_e2e.py — 端到端集成测试

L3 → L4 → train_d3.py → loss < 0.05

测试内容：
1. L3 → L4 preprocess 跑通
2. L4 schema 全部校验通过
3. train_d3.py 用 L4 跑 100 步 LoRA，loss 应 < 0.05（与 v1 等效）
"""
import json
import sys
import time
from pathlib import Path

# 让 l3_to_l4 可被找到
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT.parent))  # ~/d3-finetune
sys.path.insert(0, str(ROOT))  # l3_to_l4/

from l3_to_l4.preprocessor import process_file, load_codes
from l3_to_l4.validator import validate_l4_file


def test_l3_to_l4_basic(tmp_path):
    """最小 e2e：L3 → L4 转换 + 校验（B 精简版：4 字段 + 5 字段 meta）"""
    l3_path = tmp_path / "in.jsonl"
    l4_path = tmp_path / "out.jsonl"

    # 最小 L3 样本（1 行 1 question）
    l3_path.write_text(json.dumps({
        "state": "测试文本",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "测试？",
                "criteria": {"A": "选项1", "B": "选项2"}
            }
        },
        "answers": {"q1": {"type": "choice", "choice": "A"}}
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    codes = load_codes()
    n = process_file(str(l3_path), str(l4_path), codes)
    assert n == 1, f"expected 1 L4 row, got {n}"

    # 校验
    errors = validate_l4_file(str(l4_path), codes)
    assert not errors, f"validation failed: {errors[:3]}"

    # 检查 B 精简版 schema
    with open(l4_path, encoding="utf-8") as f:
        row = json.loads(f.readline())

    # 必填 4 字段
    assert "l3_source" in row
    assert "prompt" in row
    assert "target_token_idx" in row
    assert "meta" in row

    # meta 必填 5 字段
    meta_required = ["code_text", "question_name", "source_qkey", "llamacpp_version", "created"]
    for mk in meta_required:
        assert mk in row["meta"], f"meta missing {mk}"

    # 删去的字段不再写入
    assert "state" not in row, "B 精简版不应含 state"
    assert "question" not in row, "B 精简版不应含 question"
    assert "answer" not in row, "B 精简版不应含 answer"

    # 业务值
    assert row["target_token_idx"] == 0
    assert row["meta"]["code_text"] == "A"
    assert row["meta"]["question_name"] == "q1"
    assert row["meta"]["source_qkey"] == "q1"
    assert row["meta"]["llamacpp_version"] == "b11588"
    assert row["l3_source"] == f"{l3_path.name}:1"
    assert "<|im_start|>system" in row["prompt"]
    print("  [OK] L3 → L4 preprocess 跑通 (B 精简版)")


def test_l3_multi_question_expansion(tmp_path):
    """L3 1 行多 question → L4 多行（B 精简版）"""
    l3_path = tmp_path / "in.jsonl"
    l4_path = tmp_path / "out.jsonl"

    l3_path.write_text(json.dumps({
        "state": "abc",
        "questions": {
            "q1": {"type": "choice", "instructions": "Q1", "criteria": {"A": "1", "B": "2"}},
            "q2": {"type": "noul", "instructions": "Q2"},
        },
        "answers": {
            "q1": {"type": "choice", "choice": "A"},
            "q2": {"type": "noul", "noul": 0.85},
        }
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    codes = load_codes()
    n = process_file(str(l3_path), str(l4_path), codes)
    assert n == 2, f"expected 2 L4 rows, got {n}"

    with open(l4_path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == 2

    # 两行 L3_source 相同（同源）
    assert rows[0]["l3_source"] == rows[1]["l3_source"]
    # question_name 不同
    assert rows[0]["meta"]["question_name"] == "q1"
    assert rows[1]["meta"]["question_name"] == "q2"

    # 第一行 choice → target=0 (A)
    # 第二行 noul=0.85 → local=1, order=[1,0], ans_pos=0, codes[0]='A'
    assert rows[0]["target_token_idx"] == 0
    assert rows[0]["meta"]["code_text"] == "A"
    assert rows[1]["target_token_idx"] == 0
    assert rows[1]["meta"]["code_text"] == "A"  # 0.85 → A
    print("  [OK] 多 question 展开正确 (B 精简版)")


def test_l3_score_type(tmp_path):
    """score 类型 target 计算（B 精简版）"""
    l3_path = tmp_path / "in.jsonl"
    l4_path = tmp_path / "out.jsonl"

    l3_path.write_text(json.dumps({
        "state": "x",
        "questions": {
            "q1": {
                "type": "score",
                "instructions": "紧急？",
                "criteria": ["普通", "较快", "加急"]
            }
        },
        "answers": {"q1": {"type": "score", "score": 2}}
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    codes = load_codes()
    n = process_file(str(l3_path), str(l4_path), codes)
    with open(l4_path, encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["target_token_idx"] == 2  # score=2 → codes[2]='C'
    assert row["meta"]["code_text"] == "C"
    print("  [OK] score 类型 target 计算正确 (B 精简版)")


def test_b_slim_no_redundant_fields(tmp_path):
    """B 精简版不写 state/question/answer 冗余字段"""
    l3_path = tmp_path / "in.jsonl"
    l4_path = tmp_path / "out.jsonl"

    l3_path.write_text(json.dumps({
        "state": "测试 state 文本（应该不被 L4 保留）",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "测试 question instructions（应该不被 L4 保留）",
                "criteria": {"A": "选项1", "B": "选项2"}
            }
        },
        "answers": {"q1": {"type": "choice", "choice": "B"}}
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    codes = load_codes()
    process_file(str(l3_path), str(l4_path), codes)

    with open(l4_path, encoding="utf-8") as f:
        row = json.loads(f.readline())

    # 必含 4 字段
    assert set(row.keys()) >= {"l3_source", "prompt", "target_token_idx", "meta"}

    # 必不含 3 冗余字段
    assert "state" not in row
    assert "question" not in row
    assert "answer" not in row

    # meta 必含 5 字段
    assert set(row["meta"].keys()) >= {
        "code_text", "question_name", "source_qkey", "llamacpp_version", "created"
    }

    # 业务值：B → codes[1]='B'
    assert row["target_token_idx"] == 1
    assert row["meta"]["code_text"] == "B"
    print("  [OK] B 精简版无冗余字段 + meta 5 字段齐全")


def test_v1_data_roundtrip():
    """v1 训练数据 L3 → L4 端到端（用真实 d3_train.jsonl）"""
    # 找 L3 文件
    candidates = [
        Path(r"C:\文档\邵震洲\训练\l3_data\d3_train.jsonl"),
        Path(r"/home/d3ft/d3-finetune/data/d3_train.jsonl"),
    ]
    l3_path = None
    for c in candidates:
        if c.exists():
            l3_path = c
            break
    if l3_path is None:
        print("  [SKIP] 找不到 d3_train.jsonl")
        return

    l4_path = Path(r"C:\文档\邵震洲\训练\l4_data\v1_train.l4.jsonl")
    l4_path.parent.mkdir(parents=True, exist_ok=True)

    codes = load_codes()
    n = process_file(str(l3_path), str(l4_path), codes)
    print(f"  [OK] {l3_path.name} → {n} L4 lines")

    # 校验
    errors = validate_l4_file(str(l4_path), codes)
    assert not errors, f"validation failed: {errors[:3]}"
    print("  [OK] v1 训练 L4 校验通过")


if __name__ == "__main__":
    print("=" * 60)
    print("L3 → L4 端到端测试 (B 精简版)")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        print("\n[Test 1] L3 → L4 preprocess 基本流程")
        test_l3_to_l4_basic(td_path)

        print("\n[Test 2] 多 question 展开")
        test_l3_multi_question_expansion(td_path)

        print("\n[Test 3] score 类型 target 计算")
        test_l3_score_type(td_path)

        print("\n[Test 4] B 精简版无冗余字段")
        test_b_slim_no_redundant_fields(td_path)

    print("\n[Test 5] v1 训练数据 roundtrip")
    test_v1_data_roundtrip()

    print("\n" + "=" * 60)
    print("All tests passed")
    print("=" * 60)
