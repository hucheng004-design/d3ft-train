"""l3_to_l4/expand.py — L3 展开器

从 train_d3.py 抽离的 expand_dataset() 函数。
一行 L3 (多 question) → N 行 L3.5 (单 question)。
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def expand_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """一行 L3 → N 个 L3.5 单 question 样本

    Args:
        row: L3 的一行 {state, questions: {qname: q}, answers: {qname: a}}

    Returns:
        N 个 dict，每个含 {state, question, answer, source_qkey}：
        [
            {state, question=q1, answer=a1, source_qkey="q1"},
            {state, question=q2, answer=a2, source_qkey="q2"},
            ...
        ]
    """
    out = []
    state = row.get("state", "")
    for qname, q in row.get("questions", {}).items():
        a = row.get("answers", {}).get(qname)
        if a is None:
            raise ValueError(f"L3 schema error: question '{qname}' has no answer")
        out.append({
            "state": state,
            "question": q,
            "answer": a,
            "source_qkey": qname,
        })
    return out


def expand_dataset(jsonl_path: str) -> List[Dict[str, Any]]:
    """从 L3 jsonl 文件读所有行，展开为 L3.5 列表

    Args:
        jsonl_path: L3 文件路径

    Returns:
        展开后的 L3.5 列表（每个元素来自一行 L3 的某个 question）

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON 解析失败
    """
    p = Path(jsonl_path)
    if not p.exists():
        raise FileNotFoundError(f"L3 file not found: {p}")
    out = []
    with open(p, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at {p}:{line_num}: {e.msg}", e.doc, e.pos
                )
            out.extend(expand_row(row))
    return out
