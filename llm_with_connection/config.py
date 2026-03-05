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


# provider: "remote" or "openai"
PLANNER_PROVIDER = _env("PLANNER_PROVIDER", "remote")
EXECUTOR_PROVIDER = _env("EXECUTOR_PROVIDER", "remote")

OPENAI_API_KEY = _env("OPENAI_API_KEY", "EMPTY")

PLANNER_API_KEY = _env("PLANNER_API_KEY", OPENAI_API_KEY)
PLANNER_API_BASE = _env("PLANNER_API_BASE", "http://10.134.114.3:5000/v1")
PLANNER_MODEL_NAME = _env("PLANNER_MODEL_NAME", "models/Qwen3-4B-Instruct-2507")

EXECUTOR_API_KEY = _env("EXECUTOR_API_KEY", OPENAI_API_KEY)
EXECUTOR_API_BASE = _env("EXECUTOR_API_BASE", "http://10.134.114.3:5001/v1")
EXECUTOR_MODEL_NAME = _env("EXECUTOR_MODEL_NAME", "models/gemma3-1b-it")

HOST = _env("HOST", "localhost")
PORT = _env_i("PORT", 65432)

SYSTEM_TASK_DEFAULT = _env("SYSTEM_TASK_DEFAULT", "浣滄垬妯℃嫙")
RESET_TIME_THRESHOLD = _env_f("RESET_TIME_THRESHOLD", 5000)

# Planner/Executor cadence
PLANNER_REFRESH_INTERVAL_S = _env_f("PLANNER_REFRESH_INTERVAL_S", 10.0)
EXECUTOR_TARGET_HORIZON_S = _env_f("EXECUTOR_TARGET_HORIZON_S", 3.0)
EXECUTOR_DEFAULT_HOLD_S = _env_f("EXECUTOR_DEFAULT_HOLD_S", 1.0)
EXECUTOR_MAX_SEQUENCE_STEPS = _env_i("EXECUTOR_MAX_SEQUENCE_STEPS", 12)
EXECUTOR_MAX_SEQUENCE_TOTAL_S = _env_f("EXECUTOR_MAX_SEQUENCE_TOTAL_S", 15.0)

# Multi-frame context
LLM_CONTEXT_FRAMES = _env_i("LLM_CONTEXT_FRAMES", 1)
LLM_CONTEXT_STRIDE = _env_i("LLM_CONTEXT_STRIDE", 1)
LLM_STATE_BUFFER_MAX_FRAMES = _env_i("LLM_STATE_BUFFER_MAX_FRAMES", 200)
LLM_CONTEXT_MIN_FRAMES = _env_i("LLM_CONTEXT_MIN_FRAMES", 1)
