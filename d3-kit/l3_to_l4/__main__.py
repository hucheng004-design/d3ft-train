"""l3_to_l4/__main__.py — CLI 入口

用法:
    python -m l3_to_l4 preprocess <l3.jsonl> <l4.jsonl>
    python -m l3_to_l4 batch <l3_dir> <l4_dir> [--glob "*.jsonl"]
    python -m l3_to_l4 validate <l4.jsonl>
    python -m l3_to_l4 stats <l4.jsonl>
"""
import sys
import argparse
import json
from pathlib import Path
from collections import Counter

from .preprocessor import process_file, batch_process, load_codes
from .validator import validate_l4_file


def cmd_preprocess(args):
    codes = load_codes() if not args.codes else load_codes(args.codes)
    process_file(args.l3, args.l4, codes)


def cmd_batch(args):
    codes = load_codes() if not args.codes else load_codes(args.codes)
    batch_process(args.l3_dir, args.l4_dir, args.glob, codes)


def cmd_validate(args):
    codes = load_codes() if not args.codes else load_codes(args.codes)
    errors = validate_l4_file(args.l4, codes)
    if errors:
        print(f"[validate] FAIL: {len(errors)} errors")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    else:
        print(f"[validate] OK: {args.l4} passed all checks")


def cmd_stats(args):
    p = Path(args.l4)
    if not p.exists():
        print(f"[stats] file not found: {p}")
        sys.exit(1)
    n = 0
    qkey_counter = Counter()
    prompt_len_total = 0
    target_idx_dist = Counter()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            qkey_counter[row.get("meta", {}).get("question_name", "?")] += 1
            prompt_len_total += len(row["prompt"])
            target_idx_dist[row["target_token_idx"]] += 1
    if n == 0:
        print(f"[stats] empty file: {p}")
        return
    avg_prompt = prompt_len_total / n
    print(f"[stats] {p}")
    print(f"  total: {n}")
    print(f"  by qkey (top 10): {dict(qkey_counter.most_common(10))}")
    print(f"  avg prompt len: {avg_prompt:.0f} chars")
    print(f"  target_token_idx distribution (top 10): {dict(target_idx_dist.most_common(10))}")


def main():
    parser = argparse.ArgumentParser(
        prog="l3_to_l4",
        description="L3 (raw标注) → L4 (训练就绪) 离线转换器",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="单文件 L3 → L4")
    p_pre.add_argument("l3", help="L3 文件路径")
    p_pre.add_argument("l4", help="L4 输出文件路径")
    p_pre.add_argument("--codes", help="decision_config.json 路径（可选）")
    p_pre.set_defaults(func=cmd_preprocess)

    # batch
    p_batch = subparsers.add_parser("batch", help="批量转换")
    p_batch.add_argument("l3_dir", help="L3 目录")
    p_batch.add_argument("l4_dir", help="L4 输出目录")
    p_batch.add_argument("--glob", default="*.jsonl", help="文件匹配模式（默认 *.jsonl）")
    p_batch.add_argument("--codes", help="decision_config.json 路径（可选）")
    p_batch.set_defaults(func=cmd_batch)

    # validate
    p_val = subparsers.add_parser("validate", help="校验 L4 文件")
    p_val.add_argument("l4", help="L4 文件路径")
    p_val.add_argument("--codes", help="decision_config.json 路径（可选）")
    p_val.set_defaults(func=cmd_validate)

    # stats
    p_stats = subparsers.add_parser("stats", help="L4 文件统计")
    p_stats.add_argument("l4", help="L4 文件路径")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
