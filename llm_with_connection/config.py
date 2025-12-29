import os


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _env_f(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_i(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


# provider:
# - "remote": 走你现在的远程 OpenAI-兼容服务器（需要提供 *_API_BASE）
# - "openai": 走 OpenAI 官方 API（忽略 *_API_BASE，使用官方默认 base_url）
PLANNER_PROVIDER = _env("PLANNER_PROVIDER", "remote")
EXECUTOR_PROVIDER = _env("EXECUTOR_PROVIDER", "remote")

# Key 优先使用各自的 *_API_KEY；也可统一使用 OPENAI_API_KEY（两者都没填时保留 EMPTY 兼容旧环境）
OPENAI_API_KEY = _env("OPENAI_API_KEY", "EMPTY")

PLANNER_API_KEY = _env("PLANNER_API_KEY", OPENAI_API_KEY)
PLANNER_API_BASE = _env("PLANNER_API_BASE", "http://10.134.114.3:5000/v1")
PLANNER_MODEL_NAME = _env("PLANNER_MODEL_NAME", "models/Qwen3-4B-Instruct-2507")

EXECUTOR_API_KEY = _env("EXECUTOR_API_KEY", OPENAI_API_KEY)
EXECUTOR_API_BASE = _env("EXECUTOR_API_BASE", "http://10.134.114.3:5001/v1")
EXECUTOR_MODEL_NAME = _env("EXECUTOR_MODEL_NAME", "models/gemma3-1b-it")

HOST = _env("HOST", "localhost")
PORT = _env_i("PORT", 65432)

SYSTEM_TASK_DEFAULT = _env("SYSTEM_TASK_DEFAULT", "作战模拟")
RESET_TIME_THRESHOLD = _env_f("RESET_TIME_THRESHOLD", 5000)

# =============================
# 分层决策解耦（Planner 低频 / Executor 高频）
# =============================
# Planner 多久刷新一次“作战指令”(plan)。设大一些可以显著减少上层重复决策。
PLANNER_REFRESH_INTERVAL_S = _env_f("PLANNER_REFRESH_INTERVAL_S", 10.0)

# Executor 若支持输出动作序列，则每次推理尽量覆盖未来多少秒（server 会在时间窗内自循环执行，不必反复请求上层/下层）。
EXECUTOR_TARGET_HORIZON_S = _env_f("EXECUTOR_TARGET_HORIZON_S", 3.0)

# 单步动作默认持续时间（当 Executor 只输出单步动作、或未给 duration/hold 字段时）。
EXECUTOR_DEFAULT_HOLD_S = _env_f("EXECUTOR_DEFAULT_HOLD_S", 1.0)

# 允许的最大动作序列长度/总持续时间（防止模型输出过长导致延迟或不稳定）。
EXECUTOR_MAX_SEQUENCE_STEPS = _env_i("EXECUTOR_MAX_SEQUENCE_STEPS", 12)
EXECUTOR_MAX_SEQUENCE_TOTAL_S = _env_f("EXECUTOR_MAX_SEQUENCE_TOTAL_S", 15.0)

