from llm_with_connection import (
    EXECUTOR_API_BASE,
    EXECUTOR_API_KEY,
    EXECUTOR_MODEL_NAME,
    HOST,
    PLANNER_API_BASE,
    PLANNER_API_KEY,
    PLANNER_MODEL_NAME,
    PORT,
    RESET_TIME_THRESHOLD,
    SYSTEM_TASK_DEFAULT,
    build_langgraph_app,
    create_openai_client,
    run_socket_server,
)
from openai_key import OPENAI_API_KEY
#
# ==========================
# 在这里显式控制 Planner/Executor 走官方 OpenAI 还是远程兼容服务器
# ==========================
# provider:
# - "remote": 走远程 OpenAI-兼容服务器（使用 api_base）
# - "openai": 走 OpenAI 官方 API（忽略 api_base，只用 api_key）
#
# 你要切换模型/服务器时，改这里就行：
PLANNER_CLIENT_CFG = {
    "provider": "remote",  # "remote" | "openai"
    # 强烈建议不要把真实 key 写进代码/仓库里（容易泄露）。
    # 请改成你自己的加载方式（例如本机私有的 wow/secrets.py 或运行时输入），这里先留占位符。
    "api_key": PLANNER_API_KEY,
    "api_base": PLANNER_API_BASE,
    "model_name": PLANNER_MODEL_NAME,
    # "model_name": "gpt-5.2",
}

EXECUTOR_CLIENT_CFG = {
    "provider": "openai",  # "remote" | "openai"
    "api_key": OPENAI_API_KEY,
    "api_base": EXECUTOR_API_BASE,
    # "model_name": EXECUTOR_MODEL_NAME,
    "model_name": "gpt-5.2",
}


def main() -> None:
    # 若切到官方 OpenAI，但 key 仍是 EMPTY，则提前报错更友好
    if PLANNER_CLIENT_CFG["provider"].lower() == "openai" and (
        not PLANNER_CLIENT_CFG["api_key"] or PLANNER_CLIENT_CFG["api_key"] == "EMPTY"
    ):
        raise ValueError("PLANNER_CLIENT_CFG.provider='openai' 时必须提供有效的 api_key。")
    if EXECUTOR_CLIENT_CFG["provider"].lower() == "openai" and (
        not EXECUTOR_CLIENT_CFG["api_key"] or EXECUTOR_CLIENT_CFG["api_key"] == "EMPTY"
    ):
        raise ValueError("EXECUTOR_CLIENT_CFG.provider='openai' 时必须提供有效的 api_key。")

    planner_client = create_openai_client(
        api_key=PLANNER_CLIENT_CFG["api_key"],
        api_base=PLANNER_CLIENT_CFG["api_base"],
        provider=PLANNER_CLIENT_CFG["provider"],
        server_name="规划者(Planner)",
        check_connection=True,
    )
    executor_client = create_openai_client(
        api_key=EXECUTOR_CLIENT_CFG["api_key"],
        api_base=EXECUTOR_CLIENT_CFG["api_base"],
        provider=EXECUTOR_CLIENT_CFG["provider"],
        server_name="执行者(Executor)",
        check_connection=True,
    )

    app = build_langgraph_app(
        planner_client=planner_client,
        executor_client=executor_client,
        planner_model_name=PLANNER_CLIENT_CFG["model_name"],
        executor_model_name=EXECUTOR_CLIENT_CFG["model_name"],
    )

    run_socket_server(
        host=HOST,
        port=PORT,
        app=app,
        system_task=SYSTEM_TASK_DEFAULT,
        reset_time_threshold=RESET_TIME_THRESHOLD,
    )


if __name__ == "__main__":
    main()

