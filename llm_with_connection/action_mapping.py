import json

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _clean_json_text(s: str) -> str:
    clean = (s or "").strip()
    if "```json" in clean:
        return clean.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in clean:
        return clean.split("```", 1)[1].strip()
    return clean


def _cmd_to_action_640(cmd: dict) -> np.ndarray:
    """
    将单步控制指令 dict 转为长度=640的 float32 动作数组。
    idx 0 -> red_1, idx 1 -> red_2，每个平台 8 维里只用 0/1/2 三个:
      [0]=a0(dHeading_norm), [1]=a1(dAlt_norm), [2]=a2(dSpeed_norm)
    """
    action = np.zeros(640, dtype=np.float32)
    key_map = {
        0: cmd.get("red_1", cmd.get("red1", {})),
        1: cmd.get("red_2", cmd.get("red2", {})),
    }

    for i, c in key_map.items():
        if not isinstance(c, dict):
            continue

        turn_deg = float(c.get("turn_deg", 0.0))
        up_m = float(c.get("up_m", 0.0))
        dspeed_mps = float(c.get("dspeed_mps", 0.0))

        a0 = _clamp(turn_deg / 45.0, -1.0, 1.0)
        a1 = _clamp(up_m / 2000.0, -1.0, 1.0)
        a2 = _clamp(dspeed_mps / 150.0, -1.0, 1.0)

        base = 8 * i
        action[base + 0] = a0
        action[base + 1] = a1
        action[base + 2] = a2

    return action


def parse_llm_decision_to_action_sequence(
    llm_result_str: str,
    *,
    default_hold_s: float = 1.0,
    max_steps: int = 12,
    max_total_s: float = 15.0,
) -> list[tuple[float, np.ndarray]]:
    """
    解析 LLM 输出为“动作序列”：
    - 兼容旧格式：{"red_1": {...}, "red_2": {...}}
    - 新格式(推荐，可选)：{"steps": [{"hold_s": 1.0, "red_1": {...}, "red_2": {...}}, ...]}
      steps 字段也兼容叫 sequence；时长字段兼容 hold_s / duration_s / t

    返回: [(hold_seconds, action_640), ...]
    """
    try:
        clean = _clean_json_text(llm_result_str)
        cmd = json.loads(clean) if clean else {}
    except Exception:
        cmd = {}

    if not isinstance(cmd, dict):
        cmd = {}

    steps = cmd.get("steps", cmd.get("sequence"))
    if isinstance(steps, list) and steps:
        out: list[tuple[float, np.ndarray]] = []
        total = 0.0
        for step in steps[: max_steps if max_steps > 0 else len(steps)]:
            if not isinstance(step, dict):
                continue
            hold_s = step.get("hold_s", step.get("duration_s", step.get("t", default_hold_s)))
            try:
                hold_s_f = float(hold_s)
            except Exception:
                hold_s_f = float(default_hold_s)
            if hold_s_f <= 0:
                continue
            if total + hold_s_f > max_total_s:
                break
            out.append((hold_s_f, _cmd_to_action_640(step)))
            total += hold_s_f

        if out:
            return out

    # 旧格式：单步动作（可选支持顶层 hold）
    try:
        hold_s = float(cmd.get("hold_s", cmd.get("duration_s", default_hold_s)))
    except Exception:
        hold_s = float(default_hold_s)
    if hold_s <= 0:
        hold_s = float(default_hold_s)
    return [(hold_s, _cmd_to_action_640(cmd))]


def apply_llm_decision_to_sim(llm_result_str: str) -> np.ndarray:
    """
    兼容旧接口：返回单步长度=640的 float32 动作数组。
    若 LLM 输出了 steps/sequence，则只取第 1 步。
    """
    seq = parse_llm_decision_to_action_sequence(llm_result_str)
    return seq[0][1] if seq else np.zeros(640, dtype=np.float32)

