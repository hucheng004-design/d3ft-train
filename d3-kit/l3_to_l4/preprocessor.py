"""l3_to_l4/preprocessor.py — L3 → L4 转换主入口

完整流程：
1. 读 L3 jsonl
2. 校验 L3 schema
3. 展开（L3 → L3.5）
4. 渲染 prompt（L3.5 → L4）
5. 校验 L4 schema
6. 写 L4 jsonl
"""
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any

from .expand import expand_row
from .render import render_chat
from .target import compute_target_idx, compute_meta_code_text
from .validator import validate_l3_row, validate_l4_row


# 默认 codes 数量（d3 base 实际为 255）
DEFAULT_CODE_COUNT = 255
# 默认 llamacpp 版本
DEFAULT_LLAMACPP_VERSION = "b11588"
# 默认时区（北京时间）
CST = timezone(timedelta(hours=8))


def load_codes(codes_path: str = None) -> List[str]:
    """加载 255 个 answer codes 列表

    优先级: 显式路径 > $D3_MODEL_DIR/decision_config.json > 底座默认路径 > 合成兜底
    注意: 合成兜底与真实 decision_config 在两位字母区（第 26 位起）可能不一致，
          正式使用务必保证能读到模型目录里的 decision_config.json。
    """
    candidates = []
    if codes_path:
        candidates.append(Path(codes_path))
    env_dir = os.environ.get("D3_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "decision_config.json")
    candidates.append(Path.home() / "d3-work/models/d3-flash/decision_config.json")
    candidates.append(Path.home() / "d3-finetune/base/models/vllm-sr--d3-flash/snapshots/master/decision_config.json")
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            codes = cfg["codes"]
            if len(codes) != 255:
                raise ValueError(f"decision_config codes 数量异常: {len(codes)} @ {p}")
            return codes
    print("[load_codes] WARN: 未找到 decision_config.json，使用合成 codes（仅前 26 个与真实一致，多选项场景勿用！）",
          file=sys.stderr)
    codes = [chr(ord("A") + i) for i in range(26)]  # A-Z
    for i in range(255 - 26):
        first = i // 26  # 0..9
        second = i % 26  # 0..25
        codes.append(chr(ord("A") + first) + chr(ord("A") + second))
    return codes[:255]


def process_row(l3_row: Dict[str, Any], l3_source: str, codes: List[str]) -> List[Dict[str, Any]]:
    """单行 L3 → N 行 L4 (B 精简版: 4 字段 + 5 字段 meta)

    B 精简 L4 schema:
        l3_source: 来源追踪
        prompt: 渲染好的训练 prompt
        target_token_idx: codes 列表位置 (0..254)
        meta: {code_text, question_name, source_qkey, llamacpp_version, created}
    """
    l3_errors = validate_l3_row(l3_row)
    if l3_errors:
        raise ValueError(f"L3 validation failed: {l3_errors}")

    l4_samples = []
    for l35 in expand_row(l3_row):
        question = l35["question"]
        answer = l35["answer"]
        target_token_idx = compute_target_idx(question, answer)
        code_text = compute_meta_code_text(codes, target_token_idx)
        prompt = render_chat(l35["state"], question, codes, n_images=0)

        l4 = {
            "l3_source": l3_source,
            "prompt": prompt,
            "target_token_idx": target_token_idx,
            "meta": {
                "code_text": code_text,
                "question_name": l35["source_qkey"],
                "source_qkey": l35["source_qkey"],
                "llamacpp_version": DEFAULT_LLAMACPP_VERSION,
                "created": datetime.now(CST).isoformat(),
            },
        }

        l4_errors = validate_l4_row(l4, codes)
        if l4_errors:
            raise ValueError(f"L4 validation failed: {l4_errors}")

        l4_samples.append(l4)
    return l4_samples


def process_file(l3_path: str, l4_path: str, codes: List[str] = None, verbose: bool = True) -> int:
    """整个 L3 文件 → L4 文件

    Args:
        l3_path: L3 文件路径
        l4_path: L4 输出文件路径
        codes: codes 列表，None 则用默认
        verbose: 是否打印进度

    Returns:
        写出的 L4 行数

    Raises:
        FileNotFoundError: L3 文件不存在
        ValueError: schema 校验失败
    """
    l3_p = Path(l3_path)
    l4_p = Path(l4_path)
    if not l3_p.exists():
        raise FileNotFoundError(f"L3 file not found: {l3_p}")
    if codes is None:
        codes = load_codes()

    l4_p.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    t0 = time.time()

    with open(l3_p, encoding="utf-8") as f_in, \
         open(l4_p, "w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            l3_source = f"{l3_p.name}:{line_num}"
            try:
                l4_samples = process_row(row, l3_source, codes)
            except ValueError as e:
                raise ValueError(f"{l3_source}: {e}")
            for l4 in l4_samples:
                f_out.write(json.dumps(l4, ensure_ascii=False) + "\n")
                n_out += 1
            n_in += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"[preprocess] read {n_in} L3 lines from {l3_path}")
        print(f"[preprocess] write {n_out} L4 lines to {l4_path}")
        print(f"[preprocess] time: {elapsed:.2f}s")
    return n_out


def batch_process(l3_dir: str, l4_dir: str, glob_pattern: str = "*.jsonl", codes: List[str] = None) -> int:
    """批量转换

    Args:
        l3_dir: L3 目录
        l4_dir: L4 输出目录
        glob_pattern: 文件匹配模式
        codes: codes 列表

    Returns:
        总转换行数
    """
    from pathlib import Path
    l3_d = Path(l3_dir)
    l4_d = Path(l4_dir)
    l4_d.mkdir(parents=True, exist_ok=True)

    total = 0
    for l3_path in sorted(l3_d.glob(glob_pattern)):
        l4_path = l4_d / l3_path.name.replace(".jsonl", ".l4.jsonl")
        print(f"[batch] {l3_path.name} -> {l4_path.name}")
        n = process_file(str(l3_path), str(l4_path), codes, verbose=False)
        print(f"  -> {n} lines")
        total += n
    print(f"[batch] total {total} lines")
    return total
