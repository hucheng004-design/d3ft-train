"""export_readout.py — 从 d3 AutoModel 提取 readout 决策头，存为独立 safetensors

用法: python export_readout.py [--base 模型目录] [--out 输出路径]
说明: 正常情况下 readout.safetensors 直接复用底座原件（训练全程冻结），
     仅当合并目录里丢失该文件时才需要重导出。
"""
import os, sys, argparse
from pathlib import Path

os.environ["D3_QUANT_CONFIG"] = "none"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.environ.get(
    "D3_MODEL_DIR", str(Path.home() / "d3-work/models/d3-flash")))
ap.add_argument("--out", default=str(
    Path(os.environ.get("D3_WORK", Path.home() / "d3-work")) / "output" / "merged"
    / "merged_fp16" / "readout.safetensors"))
args = ap.parse_args()

import torch
from transformers import AutoModel
from safetensors.torch import save_file

m = AutoModel.from_pretrained(
    args.base, trust_remote_code=True, dtype=torch.bfloat16,
    attn_implementation="sdpa", low_cpu_mem_usage=True, verify="none",
)
readout = m.runtime.readout.float()  # [255, 4096] FP32
print(f"readout: {readout.shape} {readout.dtype}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
save_file({"weight": readout}, args.out)
print(f"saved -> {args.out}")
