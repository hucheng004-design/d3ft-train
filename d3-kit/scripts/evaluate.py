#!/usr/bin/env python3
"""evaluate.py — d3 决策模型评估（通用版）

两种模式:
  1. 本地 GGUF:   python evaluate.py --gguf path/to/xx-Q8_0.gguf   （自动起临时服务到 $D3_EVAL_PORT）
  2. 外部服务:   python evaluate.py --api http://host:port --api-key sk-xxx
评估集: --holdout L3 jsonl（默认 $D3_WORK/data/*.holdout.jsonl 最新）；--anti 抗遗忘 L3 可选
决策协议（llama.cpp 实测验证）:
  POST /v1/chat/completions
  content = render_body(state, question)   # State+Question+Options 三段式
  temperature=0, max_tokens=1, chat_template_kwargs={"enable_thinking": false}
  响应首个 code 字母: noul A=是 B=否; choice/score 按 Options 渲染顺序还原
报告: $D3_WORK/output/eval-<tag>/report.{txt,json}；退出码 0=PASS 2=FAIL
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORK = Path(os.environ.get("D3_WORK", Path.home() / "d3-work"))
KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))

from l3_to_l4.preprocessor import load_codes          # noqa: E402
from l3_to_l4.render import render_body               # noqa: E402

PORT = int(os.environ.get("D3_EVAL_PORT", "8291"))
API_KEY = os.environ.get("D3_API_KEY", "sk-change-me")
PASS_ACC = float(os.environ.get("D3_PASS_ACC", "0.90"))
LLAMA_BIN = os.environ.get("D3_LLAMA_DIR", str(WORK / "llama-b11588"))


def http(url, payload=None, key=None, timeout=120):
    req = urllib.request.Request(url, method="GET" if payload is None else "POST")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def decode(question, code, codes):
    if code not in codes:
        return None, False
    pos = codes.index(code)
    t = question["type"]
    if t == "noul":
        return ({0: 1, 1: 0}.get(pos), pos in (0, 1))
    if t == "choice":
        keys = list(question["criteria"].keys())
        return (keys[pos] if pos < len(keys) else None), pos < len(keys)
    if t == "score":
        n = len(question.get("criteria", []))
        return (pos if pos < n else None), pos < n
    return None, False


def eval_one(base_url, key, codes, row):
    out = []
    for qn, q in row["questions"].items():
        ans = row["answers"][qn]
        body = render_body(row["state"], q, codes)
        t0 = time.time()
        try:
            resp = http(f"{base_url}/v1/chat/completions",
                        {"messages": [{"role": "user", "content": body}],
                         "temperature": 0, "max_tokens": 1,
                         "chat_template_kwargs": {"enable_thinking": False}}, key=key)
            m = resp["choices"][0]["message"]
            raw = (m.get("content") or m.get("reasoning_content") or "").strip()
        except Exception as e:
            raw = f"__ERROR__:{e}"
        dt = time.time() - t0
        t = q["type"]
        if t == "noul":
            gold = 1 if float(ans["noul"]) >= 0.5 else 0
        elif t == "choice":
            gold = ans["choice"]
        else:
            gold = int(round(float(ans["score"])))
        pred, valid = (None, False) if raw.startswith("__ERROR__") else decode(q, raw, codes)
        out.append({"qname": qn, "type": t, "gen": raw[:8], "gold": gold,
                    "pred": pred, "ok": bool(valid and pred == gold), "sec": round(dt, 2)})
    return out


def eval_file(base_url, key, codes, path, limit, workers=4):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(eval_one, base_url, key, codes, r): i for i, r in enumerate(rows)}
        for f, i in futs.items():
            for res in f.result():
                res["row"] = i
                jobs.append(res)
    return jobs


def summarize(name, results):
    n = len(results)
    acc = sum(r["ok"] for r in results) / n if n else 0.0
    by_type = {t: {"n": sum(1 for r in results if r["type"] == t),
                   "acc": round(sum(r["ok"] for r in results if r["type"] == t)
                                / max(1, sum(1 for r in results if r["type"] == t)), 4)}
               for t in {r["type"] for r in results}}
    conf = Counter((r["gold"], r["pred"]) for r in results
                   if r["type"] == "noul" and r["pred"] in (0, 1))
    lat = [r["sec"] for r in results]
    return {"name": name, "n": n, "acc": round(acc, 4), "by_type": by_type,
            "noul_confusion": {f"gold={g},pred={p}": c for (g, p), c in sorted(conf.items())},
            "invalid_output": sum(1 for r in results if r["pred"] is None),
            "latency_p50": round(statistics.median(lat), 2) if lat else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", help="本地 GGUF（模式1）")
    ap.add_argument("--api", help="已运行服务 base_url（模式2，如 http://127.0.0.1:8283）")
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--holdout", help="评估集 L3 jsonl")
    ap.add_argument("--anti", default="", help="抗遗忘 L3 jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default=time.strftime("eval-%m%d%H%M"))
    args = ap.parse_args()

    if not args.api and not args.gguf:
        args.gguf = ""
        cands = sorted((WORK / "output").glob("merged-*/d3-*-Q8_0.gguf"), key=os.path.getmtime)
        if cands:
            args.gguf = str(cands[-1])
    if not args.api and not args.gguf:
        print("[eval] FAIL: 没有 --api/--gguf，且 output/ 下未找到 Q8 GGUF")
        sys.exit(1)

    holdout = args.holdout
    if not holdout:
        cands = sorted((WORK / "data").glob("*.holdout.jsonl"), key=os.path.getmtime)
        holdout = str(cands[-1]) if cands else ""
    if not holdout or not Path(holdout).exists():
        print("[eval] FAIL: 找不到 holdout 文件（先跑 validate_data.sh）")
        sys.exit(1)

    codes = load_codes()
    base_url = args.api.rstrip("/") if args.api else f"http://127.0.0.1:{PORT}"
    out_dir = WORK / "output" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = None
    try:
        if args.gguf:
            print(f"[eval] 启动验证服务: {Path(args.gguf).name} @ {base_url}")
            env = dict(os.environ, LD_LIBRARY_PATH=LLAMA_BIN)
            logf = open(out_dir / "server.log", "w")
            proc = subprocess.Popen(
                [str(Path(LLAMA_BIN) / "llama-server"), "-m", args.gguf,
                 "-ngl", "99", "-c", "8192", "-fa", "on", "--jinja",
                 "--port", str(PORT), "--host", "127.0.0.1", "--api-key", args.api_key],
                stdout=logf, stderr=subprocess.STDOUT, env=env)
            for _ in range(60):
                time.sleep(5)
                try:
                    http(f"{base_url}/health", timeout=5)
                    break
                except Exception:
                    pass
            else:
                subprocess.run(f"tail -20 {(out_dir / 'server.log')}", shell=True)
                print("[eval] FAIL: 服务未就绪")
                sys.exit(1)
            key = args.api_key
        else:
            http(f"{base_url}/health", timeout=5)
            key = args.api_key
        print("[eval] 服务就绪，开始评估...")

        rep = {"holdout": summarize("holdout", eval_file(base_url, key, codes, holdout, args.limit))}
        if args.anti and Path(args.anti).exists():
            rep["anti"] = summarize("anti", eval_file(base_url, key, codes, args.anti, args.limit))

        acc = rep["holdout"]["acc"]
        anti = rep.get("anti", {}).get("acc")
        verdict = acc >= PASS_ACC and (anti is None or anti >= PASS_ACC)
        rep["threshold"] = PASS_ACC
        rep["verdict"] = "PASS" if verdict else "FAIL"
        (out_dir / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

        lines = [f"# 评估报告 {args.tag}"]
        for part in ("holdout", "anti"):
            if part in rep:
                r = rep[part]
                lines += [f"{part}: acc={r['acc']:.2%} (n={r['n']}) 类型={r['by_type']}",
                          f"  noul混淆={r['noul_confusion']} 非法输出={r['invalid_output']} p50={r['latency_p50']}s"]
        lines.append(f"判定: {rep['verdict']}（阈值 ≥{PASS_ACC}）")
        (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        sys.exit(0 if verdict else 2)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            print("[eval] 临时验证服务已停止")


if __name__ == "__main__":
    main()
