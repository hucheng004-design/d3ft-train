"""merge_lora_to_gguf.py — 把 LoRA adapter 合并到 4bit backbone，再导出 GGUF

步骤：
1. 加载 4bit d3 backbone（同 train 方式）
2. 加载 LoRA adapter（peft）
3. merge_and_unload 合并到 backbone
4. 保存为 fp16 safetensors
5. 用 llama.cpp convert_hf_to_gguf.py 转 GGUF F16
6. 用 llama.cpp llama-quantize 量化 Q8_0

输入：~/d3-finetune/output/aug-save-2026-10-15/adapter-final
输出：~/d3-finetune/output/merged/social-security-d3-flash-Q8_0.gguf
"""
import os, sys, json, gc
from pathlib import Path

# 启用 4bit 环境（必须与训练一致）
os.environ["D3_QUANT_CONFIG"] = "none"  # merge 时切回 bf16（需要 fp16 weights）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default=os.environ.get(
        "D3_MODEL_DIR", str(Path.home() / "d3-work/models/d3-flash")))
    p.add_argument("--out", default=str(
        Path(os.environ.get("D3_WORK", Path.home() / "d3-work")) / "output" / "merged"))
    p.add_argument("--no-quant", action="store_true", help="不量化，输出 F16")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 加载 d3 bf16（merge 需要 fp16 权重）
    print("[load] d3 bf16 backbone...")
    model = AutoModel.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, verify="none",
    )
    if hasattr(model.backbone, "visual"):
        model.backbone.visual.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
    print(f"[mem] {torch.cuda.memory_allocated()/2**30:.1f}GB")

    # 2. 加载 adapter 并 merge
    print(f"[load] adapter from {args.adapter}...")
    peft_model = PeftModel.from_pretrained(model.backbone, args.adapter, torch_dtype=torch.bfloat16)
    print("[merge] merging LoRA into backbone...")
    merged = peft_model.merge_and_unload()
    print(f"[merge] done. merged type: {type(merged).__name__}")

    # 3. 保存为 fp16 safetensors
    fp16_dir = out / "merged_fp16"
    if fp16_dir.exists():
        import shutil; shutil.rmtree(fp16_dir)
    fp16_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save fp16] -> {fp16_dir}")
    # 复制 config/tokenizer 必需文件
    import shutil
    for f in ("config.json", "decision_config.json", "chat_template.jinja",
              "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
              "preprocessor_config.json", "special_tokens_map.json"):
        src = Path(args.base) / f
        if src.exists():
            shutil.copy2(src, fp16_dir / f)
    # 保存 backbone 权重
    merged.save_pretrained(str(fp16_dir), safe_serialization=True, max_shard_size="5GB")
    print(f"[save fp16] done. files: {sorted(p.name for p in fp16_dir.iterdir())}")


if __name__ == "__main__":
    main()
