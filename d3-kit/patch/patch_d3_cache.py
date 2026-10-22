#!/usr/bin/env python3
"""patch_d3_cache.py — d3 训练前置补丁（幂等，可重复执行）

原理: transformers trust_remote_code 会把模型目录的 .py 复制到
  ~/.cache/huggingface/modules/transformers_modules/<名字>/ 下并从那里加载。
训练用 4bit 量化需要给这份缓存代码打两个补丁：
  1. modeling_d3.py  — 接受 quantization_config 关键字
  2. d3_runtime.py   — 读 D3_QUANT_CONFIG=4bit 环境变量，按 NF4 量化加载骨干

用法:
  python patch_d3_cache.py --model-dir /path/to/d3-flash            # 检查+补丁
  python patch_d3_cache.py --model-dir ... --check                  # 只检查
补丁缺失且缓存为空时，脚本自动用 --model-dir 做一次 CPU 预加载以生成缓存（需内存≥20GB）。
"""
import argparse
import os
import sys
from pathlib import Path

CACHE_ROOT = Path.home() / ".cache/huggingface/modules/transformers_modules"

PATCH_MODELING = (
    '        if kwargs:\n            raise TypeError(',
    '        kwargs.pop("quantization_config", None)\n        if kwargs:\n            raise TypeError(',
)
PATCH_RUNTIME_TARGET = '''        self.backbone = backbone_class(self.backbone_type).from_pretrained(
            str(self.root),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": str(self.device)},
        )'''
PATCH_RUNTIME_NEW = '''        import os as _os
        _qc = _os.environ.get("D3_QUANT_CONFIG", "none")
        _kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": str(self.device)})
        if _qc == "4bit":
            from transformers import BitsAndBytesConfig
            _kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        self.backbone = backbone_class(self.backbone_type).from_pretrained(str(self.root), **_kwargs)'''


def find_cache_files():
    if not CACHE_ROOT.exists():
        return []
    return list(CACHE_ROOT.glob("**/d3_runtime.py"))


def warmup(model_dir: str):
    print("[patch] 缓存为空，做一次 CPU 预加载生成 trust_remote_code 缓存（约 1-3 分钟）...")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from transformers import AutoModel
    AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16,
                              attn_implementation="sdpa", low_cpu_mem_usage=True, verify="none")
    print("[patch] 预加载完成")


def patch_file(fp: Path, target: str, repl: str, marker: str, name: str) -> bool:
    s = fp.read_text(encoding="utf-8")
    if marker in s:
        print(f"[patch] {name} 已是最新: {fp.parent.name}")
        return True
    if target not in s:
        print(f"[patch] ERR: {name} 目标代码段未找到: {fp}")
        return False
    fp.write_text(s.replace(target, repl, 1), encoding="utf-8")
    print(f"[patch] 已补丁 {name}: {fp.parent.name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="d3-flash 模型目录")
    ap.add_argument("--check", action="store_true", help="只检查不修改")
    args = ap.parse_args()

    files = find_cache_files()
    if not files:
        if args.check:
            print("[patch] 缓存为空（尚未做 trust_remote_code 加载）")
            sys.exit(2)
        warmup(args.model_dir)
        files = find_cache_files()
        if not files:
            print("[patch] ERR: 预加载后仍未找到缓存，检查模型目录是否正确")
            sys.exit(1)

    ok = True
    for fp in files:
        if args.check:
            s = fp.read_text(encoding="utf-8")
            need1 = 'kwargs.pop("quantization_config"' not in (fp.parent / "modeling_d3.py").read_text(encoding="utf-8") if (fp.parent / "modeling_d3.py").exists() else True
            need2 = "D3_QUANT_CONFIG" not in s
            if need1 or need2:
                print(f"[patch] 缺补丁: {fp.parent.name} （去掉 --check 执行补丁）")
                ok = False
            else:
                print(f"[patch] OK: {fp.parent.name}")
            continue
        m = fp.parent / "modeling_d3.py"
        if m.exists():
            ok &= patch_file(m, *PATCH_MODELING, 'kwargs.pop("quantization_config"', "modeling_d3")
        ok &= patch_file(fp, PATCH_RUNTIME_TARGET, PATCH_RUNTIME_NEW, "D3_QUANT_CONFIG", "d3_runtime")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
