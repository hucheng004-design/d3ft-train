#!/usr/bin/env python3
"""validate_data.py — L3 数据校验 → holdout 切分 → L4 转换（通用版）

输入: L3 jsonl {state, questions, answers}
输出: $D3_WORK/data/<stem>.train.jsonl / .holdout.jsonl
      $D3_WORK/l4_data/<stem>.train.l4.jsonl / .holdout.l4.jsonl
退出码 0=通过；1=有错（不进入训练）
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

WORK = Path(os.environ.get("D3_WORK", Path.home() / "d3-work"))
KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from l3_to_l4.validator import validate_l3_row, validate_l4_file  # noqa: E402
from l3_to_l4.preprocessor import load_codes, process_file        # noqa: E402


def noul_label(v: float) -> str:
    if v >= 0.75:
        return "是(1)"
    if v <= 0.25:
        return "否(0)"
    return "不确定(0.5)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="L3 jsonl 路径")
    ap.add_argument("--holdout-ratio", type=float, default=float(os.environ.get("D3_HOLDOUT_RATIO", "0.05")))
    ap.add_argument("--use-all", action="store_true", help="训练改用全量（holdout 仍切出供评估）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    l3_path = Path(args.data)
    if not l3_path.is_absolute():
        l3_path = WORK / l3_path
    if not l3_path.exists():
        print(f"[validate] FAIL: 文件不存在: {l3_path}")
        sys.exit(1)

    codes = load_codes()

    rows, errors = [], []
    with open(l3_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON: {e}")
    for i, row in enumerate(rows, 1):
        for e in validate_l3_row(row):
            errors.append(f"line {i}: {e}")
    if errors:
        print(f"[validate] FAIL: {len(errors)} 个错误（前 20 条）:")
        for e in errors[:20]:
            print(f"  {e}")
        sys.exit(1)
    print(f"[validate] schema 通过: {len(rows)} 行")

    qtype, noul_dist, sigs, qnames = Counter(), Counter(), Counter(), Counter()
    for row in rows:
        sigs[json.dumps(row.get("state"), ensure_ascii=False, sort_keys=True)] += 1
        for qn, q in row["questions"].items():
            qtype[q.get("type", "?")] += 1
            qnames[qn] += 1
            a = row["answers"].get(qn, {})
            if q.get("type") == "noul" and "noul" in a:
                noul_dist[noul_label(float(a["noul"]))] += 1
    dup = sum(c - 1 for c in sigs.values() if c > 1)
    print(f"[validate] 样本 {len(rows)} | question {sum(qtype.values())} | 类型 {dict(qtype)} | 规则 {dict(qnames)}")
    print(f"[validate] noul 标签分布 {dict(noul_dist)} | 重复 state {dup}")

    rng = random.Random(args.seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_hold = max(2, int(len(rows) * args.holdout_ratio))
    n_hold = min(n_hold, len(rows) // 2)  # 保证 train 不少于 holdout
    hold = set(idx[:n_hold])
    train_rows = [r for i, r in enumerate(rows) if i not in hold]
    hold_rows = [r for i, r in enumerate(rows) if i in hold]
    print(f"[validate] 切分: train={len(train_rows)} holdout={len(hold_rows)}")

    stem = l3_path.stem
    train_path = WORK / "data" / f"{stem}.train.jsonl"
    hold_path = WORK / "data" / f"{stem}.holdout.jsonl"
    for p, rs in ((train_path, train_rows), (hold_path, hold_rows)):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.use_all:
        with open(train_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[validate] --use-all: 训练改用全量 {len(rows)} 行")

    ok = True
    for src in (train_path, hold_path):
        l4 = WORK / "l4_data" / f"{src.stem}.l4.jsonl"
        n = process_file(str(src), str(l4), codes)
        errs = validate_l4_file(str(l4), codes)
        if errs:
            ok = False
            print(f"[validate] L4 FAIL ({src.stem}): {len(errs)} 个错误（前 10）:")
            for e in errs[:10]:
                print(f"  {e}")
        else:
            print(f"[validate] L4 OK ({src.stem}): {n} 行 -> {l4.name}")
    if not ok:
        sys.exit(1)
    print(f"[validate] PASS — 训练文件: {train_path.relative_to(WORK)} | holdout: {hold_path.relative_to(WORK)}")


if __name__ == "__main__":
    main()
