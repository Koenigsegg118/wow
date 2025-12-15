import json

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def apply_llm_decision_to_sim(llm_result_str: str) -> np.ndarray:
    """
    返回长度=640的 float32 动作数组。
    idx 0 -> red_1, idx 1 -> red_2，每个平台 8 维里只用 0/1/2 三个:
      [0]=a0(dHeading_norm), [1]=a1(dAlt_norm), [2]=a2(dSpeed_norm)
    """
    action = np.zeros(640, dtype=np.float32)

    try:
        clean = llm_result_str.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].strip()

        cmd = json.loads(clean) if clean else {}
    except Exception:
        cmd = {}

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

