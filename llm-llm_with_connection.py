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


def main() -> None:
    planner_client = create_openai_client(
        api_key=PLANNER_API_KEY,
        api_base=PLANNER_API_BASE,
        server_name="规划者(Planner)",
        check_connection=True,
    )
    executor_client = create_openai_client(
        api_key=EXECUTOR_API_KEY,
        api_base=EXECUTOR_API_BASE,
        server_name="执行者(Executor)",
        check_connection=True,
    )

    app = build_langgraph_app(
        planner_client=planner_client,
        executor_client=executor_client,
        planner_model_name=PLANNER_MODEL_NAME,
        executor_model_name=EXECUTOR_MODEL_NAME,
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

