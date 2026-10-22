"""l3_to_l4 — L3 → L4 转换器（离线 CLI）

用法:
    python -m l3_to_l4 preprocess <l3.jsonl> <l4.jsonl>
    python -m l3_to_l4 batch <l3_dir> <l4_dir> [--glob "*.jsonl"]
    python -m l3_to_l4 validate <l4.jsonl>
    python -m l3_to_l4 stats <l4.jsonl>
"""
__version__ = "0.2.0"

# B 精简版 (2026-10-18) - 删 state/question/answer, 保留 l3_source/prompt/target_token_idx/meta
from .expand import expand_row, expand_dataset
from .render import render_chat, render_options, option_order, describe
from .target import compute_target_idx, compute_meta_code_text
from .validator import validate_l3_row, validate_l4_row, validate_l4_file
from .preprocessor import process_row, process_file, batch_process, load_codes

# 兼容旧导入名
preprocess = process_row
preprocess_file = process_file
