"""l3_to_l4/target.py — target token 索引计算

从 train_d3.py Collator 抽离的 target_token 计算逻辑。
"""
from typing import Dict, Any, List
from .render import option_order


def compute_target_idx(question: Dict, answer: Dict) -> int:
    """根据 question 和 answer 计算 target_token_idx (0..254 codes 列表索引)

    Args:
        question: 单个 question 配置
        answer: 对应 answer

    Returns:
        codes 列表中的位置索引（不是真实 vocab token id）

    Raises:
        ValueError: question/answer 类型不匹配或 answer 字段缺失
    """
    order = option_order(question)
    n = len(order)
    q_type = question.get("type")
    a_type = answer.get("type")

    if q_type != a_type:
        raise ValueError(
            f"Type mismatch: question.type={q_type}, answer.type={a_type}"
        )

    if q_type == "choice":
        if "choice" not in answer:
            raise ValueError("choice question requires answer.choice")
        keys = list(question["criteria"].keys())
        try:
            local = keys.index(answer["choice"])
        except ValueError:
            raise ValueError(
                f"answer.choice='{answer['choice']}' not in criteria keys {keys}"
            )
        ans_pos = order[local] if local < len(order) else 0

    elif q_type == "noul":
        if "noul" not in answer:
            raise ValueError("noul question requires answer.noul")
        noul_val = float(answer["noul"])
        if not (0.0 <= noul_val <= 1.0):
            raise ValueError(f"answer.noul={noul_val} not in [0, 1]")
        local = 1 if noul_val >= 0.5 else 0
        ans_pos = order[local]  # noul order = [1, 0]

    elif q_type == "score":
        if "score" not in answer:
            raise ValueError("score question requires answer.score")
        score_val = int(round(float(answer["score"])))
        local = max(0, min(score_val, n - 1))
        ans_pos = order[local]

    else:
        raise ValueError(f"Unknown question type: {q_type}")

    if not (0 <= ans_pos < 255):
        raise ValueError(f"target_token_idx {ans_pos} out of range [0, 255)")
    return ans_pos


def compute_meta_code_text(codes: List[str], target_token_idx: int) -> str:
    """从 codes 列表和 target_token_idx 取出对应的 code 字符串

    Args:
        codes: d3 的 codes 列表
        target_token_idx: 0..254

    Returns:
        code 字符串，如 "A", "B", ..., "JT"
    """
    if not (0 <= target_token_idx < len(codes)):
        raise ValueError(
            f"target_token_idx {target_token_idx} out of range [0, {len(codes)})"
        )
    return codes[target_token_idx]
