"""l3_to_l4/validator.py — L3 + L4 schema 校验器

L3 输入校验：
- state 长度合理
- questions 与 answers key 严格对齐
- 每 question.type 与 answer 类型匹配
- choice criteria key 唯一，answer.choice 存在
- noul.noul ∈ [0, 1]
- score.score 是整数 ∈ [0, n-1]

L4 输出校验：
- prompt 格式正确（im_start 标签完整）
- target_token_idx ∈ [0, 255)
- meta.code_text == codes[target_token_idx]
- 各种字段类型正确
"""
from typing import Dict, List, Any


def validate_l3_row(row: Dict[str, Any]) -> List[str]:
    """返回错误列表（空=通过）"""
    errors = []
    if "state" not in row:
        errors.append("missing 'state' field")
    else:
        state = row["state"]
        if not isinstance(state, (str, dict, list)):
            errors.append(f"state must be str/dict/list, got {type(state).__name__}")

    if "questions" not in row:
        errors.append("missing 'questions' field")
        return errors
    if "answers" not in row:
        errors.append("missing 'answers' field")
        return errors

    q_dict = row["questions"]
    a_dict = row["answers"]
    if not isinstance(q_dict, dict):
        errors.append(f"questions must be dict, got {type(q_dict).__name__}")
        return errors
    if not isinstance(a_dict, dict):
        errors.append(f"answers must be dict, got {type(a_dict).__name__}")
        return errors

    q_keys = set(q_dict.keys())
    a_keys = set(a_dict.keys())
    if q_keys != a_keys:
        errors.append(f"questions/answers key mismatch: {sorted(q_keys ^ a_keys)}")

    for qn, q in q_dict.items():
        a = a_dict.get(qn)
        if a is None:
            errors.append(f"{qn}: missing answer")
            continue
        if not isinstance(q, dict):
            errors.append(f"{qn}: question must be dict, got {type(q).__name__}")
            continue
        if not isinstance(a, dict):
            errors.append(f"{qn}: answer must be dict, got {type(a).__name__}")
            continue

        q_type = q.get("type")
        a_type = a.get("type")
        if q_type != a_type:
            errors.append(f"{qn}: type mismatch q={q_type} a={a_type}")

        if q_type == "choice":
            if "criteria" not in q or not isinstance(q["criteria"], dict):
                errors.append(f"{qn}: choice requires criteria dict")
            else:
                keys = list(q["criteria"].keys())
                if len(keys) != len(set(keys)):
                    errors.append(f"{qn}: choice criteria has duplicate keys")
                if "choice" not in a:
                    errors.append(f"{qn}: choice answer missing 'choice' field")
                elif a["choice"] not in keys:
                    errors.append(
                        f"{qn}: answer.choice='{a['choice']}' not in {keys}"
                    )
        elif q_type == "noul":
            if "noul" not in a:
                errors.append(f"{qn}: noul answer missing 'noul' field")
            else:
                v = float(a["noul"])
                if not (0.0 <= v <= 1.0):
                    errors.append(f"{qn}: noul={v} not in [0, 1]")
        elif q_type == "score":
            if "score" not in a:
                errors.append(f"{qn}: score answer missing 'score' field")
            else:
                try:
                    s = int(round(float(a["score"])))
                except (ValueError, TypeError):
                    errors.append(f"{qn}: score={a['score']} not numeric")
                    continue
                n = len(q.get("criteria", []))
                if n == 0:
                    errors.append(f"{qn}: score requires non-empty criteria array")
                elif not (0 <= s < n):
                    errors.append(
                        f"{qn}: score={s} out of range [0, {n})"
                    )
        else:
            errors.append(f"{qn}: unknown type '{q_type}'")

        if "instructions" not in q or not q["instructions"].strip():
            errors.append(f"{qn}: missing/empty 'instructions'")

    # state 长度（避免 OOM）
    state = row.get("state", "")
    if isinstance(state, str) and len(state) > 4000:
        errors.append(f"state too long: {len(state)} chars (max 4000)")

    return errors


def validate_l4_row(row: Dict[str, Any], codes: List[str]) -> List[str]:
    """返回 L4 行错误列表（空=通过）

    B 精简 L4 schema 必填字段:
        l3_source, prompt, target_token_idx, meta
    meta 内必填:
        code_text, question_name, source_qkey, llamacpp_version, created
    """
    errors = []

    # 必填字段（B 精简版 4 字段）
    required = ["l3_source", "prompt", "target_token_idx", "meta"]
    for k in required:
        if k not in row:
            errors.append(f"missing '{k}' field")

    if "l3_source" in row:
        s = row["l3_source"]
        if not isinstance(s, str) or ":" not in s:
            errors.append(f"l3_source must match 'filename.jsonl:line_number', got '{s}'")

    if "target_token_idx" in row:
        idx = row["target_token_idx"]
        if not isinstance(idx, int) or not (0 <= idx < 255):
            errors.append(f"target_token_idx={idx} not in [0, 255)")

    if "prompt" in row:
        p = row["prompt"]
        if not isinstance(p, str):
            errors.append(f"prompt must be str, got {type(p).__name__}")
        else:
            if not p.startswith("<|im_start|>system\n"):
                errors.append("prompt must start with '<|im_start|>system\\n'")
            if "<|im_end|>\n<|im_start|>user\n" not in p:
                errors.append("prompt missing user segment")
            if "<|im_start|>assistant\n" not in p:
                errors.append("prompt missing assistant segment")
            if not p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
                errors.append(f"prompt must end with assistant empty segment, got tail='{p[-30:]!r}'")

    if "meta" in row:
        meta = row["meta"]
        if not isinstance(meta, dict):
            errors.append(f"meta must be dict, got {type(meta).__name__}")
        else:
            # meta 必填 5 字段
            meta_required = ["code_text", "question_name", "source_qkey", "llamacpp_version", "created"]
            for mk in meta_required:
                if mk not in meta:
                    errors.append(f"meta missing '{mk}' field")

    if "meta" in row and "code_text" in row["meta"] and "target_token_idx" in row:
        idx = row["target_token_idx"]
        if 0 <= idx < len(codes):
            expected_code = codes[idx]
            actual_code = row["meta"]["code_text"]
            if expected_code != actual_code:
                errors.append(
                    f"meta.code_text='{actual_code}' != codes[{idx}]='{expected_code}'"
                )

    return errors


def validate_l4_file(l4_path: str, codes: List[str]) -> List[str]:
    """校验整个 L4 文件，返回所有错误（带行号）"""
    import json
    from pathlib import Path

    p = Path(l4_path)
    if not p.exists():
        return [f"file not found: {p}"]

    all_errors = []
    with open(p, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {line_num}: invalid JSON: {e}")
                continue
            row_errors = validate_l4_row(row, codes)
            for err in row_errors:
                all_errors.append(f"line {line_num}: {err}")
    return all_errors
