"""d3-flash LoRA fine-tuning for social security domain.
基于 l3_to_l4 转换器生成的 L4 jsonl 训练（不渲染 prompt）。
"""
import os, json, time, argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# 复用 l3_to_l4 的工具（不重新实现）
_KIT = Path(__file__).resolve().parent.parent
if str(_KIT) not in sys.path:
    sys.path.insert(0, str(_KIT))
from l3_to_l4.validator import validate_l4_row
from l3_to_l4.preprocessor import load_codes

BASE = Path(os.environ.get(
    "D3_MODEL_DIR",
    os.environ.get("D3_BASE", str(Path.home() / "d3-work/models/d3-flash"))))


def load_l4_dataset(l4_path):
    """从 L4 jsonl 读所有行（已渲染好的训练样本）

    L4 schema: {l3_source, state, question, answer, prompt, target_token_idx, meta}
    """
    out = []
    with open(l4_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(row)
    return out


class Collator:
    def __init__(self, tokenizer, codes):
        self.tok = tokenizer
        self.codes = codes
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch):
        prompts, lengths, targets = [], [], []
        for s in batch:
            # L4 已有 prompt（已渲染），直接用
            prompt = s["prompt"]
            # target_token_idx 是 codes 列表索引（不是真实 vocab id）
            target_token_idx = s["target_token_idx"]
            ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
            prompts.append(ids)
            lengths.append(len(ids))
            targets.append(target_token_idx)
        ml = max(lengths)
        input_ids = torch.tensor([p + [self.pad_id] * (ml - len(p)) for p in prompts], dtype=torch.long)
        am = torch.tensor([[1] * l + [0] * (ml - l) for l in lengths], dtype=torch.long)
        pos = torch.tensor([l - 1 for l in lengths], dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": am, "position_ids": pos, "target_ids": targets}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="L4 jsonl 路径（已渲染）")
    p.add_argument("--base", default=str(BASE))
    p.add_argument("--out", default=str(Path.home() / "d3-finetune/output/train"))
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--save_steps", type=int, default=50, help="每 N 步保存 adapter 检查点")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # SIGINT handler: 优雅退出保存 adapter
    import signal
    _SAVE_DIR = [None]
    _BB = [None]
    def _sigint_handler(sig, frame):
        print("\n[sigint] saving adapter...")
        if _BB[0] is not None and _SAVE_DIR[0] is not None:
            _BB[0].save_pretrained(_SAVE_DIR[0])
            print(f"[sigint] saved -> {_SAVE_DIR[0]}")
        sys.exit(0)
    signal.signal(signal.SIGINT, _sigint_handler)

    torch.manual_seed(args.seed)
    device = "cuda"

    # 4bit 加载（必须有 D3_QUANT_CONFIG=4bit + verify=none + patched d3 源码）
    assert os.environ.get("D3_QUANT_CONFIG") == "4bit", (
        "must set D3_QUANT_CONFIG=4bit env var + patch d3 base source code"
    )

    print(f"[load] {args.base}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    cfg = json.loads((Path(args.base) / "decision_config.json").read_text(encoding="utf-8"))
    for c, t in zip(cfg["codes"], cfg["token_ids"]):
        if tok(c, add_special_tokens=False)["input_ids"] != [t]:
            raise ValueError(f"code {c} mismatch")

    model = AutoModel.from_pretrained(
        args.base, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, verify="none",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ),
    )
    model = prepare_model_for_kbit_training(model)
    readout = model.runtime.readout
    print(f"[load] readout: {readout.shape} {readout.dtype}")

    if hasattr(model.backbone, "visual"):
        for n_, p_ in model.backbone.named_parameters():
            if "visual" in n_.lower() or "vision" in n_.lower():
                p_.requires_grad = False
    backbone = model.backbone
    backbone.gradient_checkpointing_enable()

    lc = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.FEATURE_EXTRACTION, bias="none",
    )
    backbone = get_peft_model(backbone, lc)
    backbone.print_trainable_parameters()
    print(f"[load] ok in {time.time()-t0:.1f}s")

    # 加载 L4 数据
    data = load_l4_dataset(args.data)
    print(f"[data] loaded {len(data)} L4 samples")

    # Collator + DataLoader
    codes = load_codes()
    collator = Collator(tok, codes)
    dl = DataLoader(data, batch_size=args.bs, shuffle=True, collate_fn=collator, num_workers=0)

    optim = torch.optim.AdamW([p for p in backbone.parameters() if p.requires_grad], lr=args.lr)
    sched = get_cosine_schedule_with_warmup(optim, 10, args.max_steps)
    backbone.train()

    _SAVE_DIR[0] = str(out / "adapter-final")
    _BB[0] = backbone

    logf = open(out / "train.log", "w", buffering=1)
    step = 0
    while step < args.max_steps:
        for b in dl:
            if step >= args.max_steps:
                break
            ii = b["input_ids"].to(device)
            am = b["attention_mask"].to(device)
            po = b["position_ids"].to(device)
            tg = b["target_ids"].to(device)
            o = backbone(input_ids=ii, attention_mask=am)
            h = o.last_hidden_state
            idx = po.view(-1, 1, 1).expand(-1, 1, h.size(-1))
            last = h.gather(1, idx).squeeze(1)
            logits = last.float() @ readout.to(device).T.float() / cfg.get("temperature", 1.0)
            loss = F.cross_entropy(logits, tg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in backbone.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            with torch.no_grad():
                acc = (logits.argmax(-1) == tg).float().mean().item()
            mem = torch.cuda.max_memory_allocated() / 2**30
            msg = f"step {step+1}/{args.max_steps} loss={loss.item():.4f} acc={acc:.2f} mem={mem:.1f}GB"
            print(msg)
            logf.write(msg + "\n")
            # checkpoint
            if step % args.save_steps == 0:
                ckpt = out / f"adapter-step{step}"
                backbone.save_pretrained(str(ckpt))
                tok.save_pretrained(str(ckpt))
                print(f"[save] adapter -> {ckpt}")
            step += 1

    # 最终保存
    backbone.save_pretrained(_SAVE_DIR[0])
    tok.save_pretrained(_SAVE_DIR[0])
    print(f"[save final] adapter -> {_SAVE_DIR[0]}")
    logf.close()
    print("[done]")


if __name__ == "__main__":
    main()
