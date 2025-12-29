import socket
import threading
from queue import Full, Queue

import numpy as np

from .action_mapping import apply_llm_decision_to_sim, parse_llm_decision_to_action_sequence
from .config import (
    EXECUTOR_DEFAULT_HOLD_S,
    EXECUTOR_MAX_SEQUENCE_STEPS,
    EXECUTOR_MAX_SEQUENCE_TOTAL_S,
    EXECUTOR_TARGET_HORIZON_S,
    PLANNER_REFRESH_INTERVAL_S,
)
from .sim_translation import translate_sim_data_to_llm_context
from .socket_protocol import StateReceiver, send_reset_instruction, send_status_data


def run_socket_server(
    *,
    host: str,
    port: int,
    app,
    system_task: str,
    reset_time_threshold: float = 5000,
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Socket Server listening on {host}:{port}")
    print("等待仿真环境连接...")

    connection, address = server.accept()
    print(f"已连接 {address}")

    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

    receiver = StateReceiver()

    last_action = np.zeros(640, dtype=np.float32)
    action_lock = threading.Lock()

    # 动作序列缓冲：[(end_sim_time, action_640), ...]
    action_timeline: list[tuple[float, np.ndarray]] = []

    # Planner 缓存（上层低频）
    current_plan: str | None = None
    last_plan_time_sim: float = -1e30

    state_q: "Queue[tuple[float, np.ndarray]]" = Queue(maxsize=1)

    def push_latest_state(item) -> None:
        try:
            state_q.put_nowait(item)
            return
        except Full:
            pass
        try:
            _ = state_q.get_nowait()
        except Exception:
            pass
        try:
            state_q.put_nowait(item)
        except Exception:
            pass

    def llm_worker() -> None:
        while True:
            sim_time, sim_data = state_q.get()

            llm_context = translate_sim_data_to_llm_context(sim_time, sim_data)
            base_inputs = {"task": system_task, "dynamic_context": llm_context}

            try:
                # 兼容旧 app：没有 invoke_planner/invoke_executor 时，仍走一次 full invoke
                has_split = hasattr(app, "invoke_planner") and hasattr(app, "invoke_executor")

                nonlocal current_plan, last_plan_time_sim, action_timeline

                if has_split:
                    # 1) Planner 低频刷新 plan
                    need_refresh_plan = (current_plan is None) or (
                        sim_time - last_plan_time_sim >= PLANNER_REFRESH_INTERVAL_S
                    )
                    if need_refresh_plan:
                        p_state = app.invoke_planner(base_inputs)
                        plan = p_state.get("plan")
                        if isinstance(plan, str) and plan.strip():
                            current_plan = plan.strip()
                            last_plan_time_sim = sim_time

                    # 2) 若已有动作覆盖未来一段时间，则不必频繁调用 Executor
                    with action_lock:
                        last_end = action_timeline[-1][0] if action_timeline else -1e30
                    if last_end >= sim_time + EXECUTOR_TARGET_HORIZON_S:
                        continue

                    # 3) Executor 基于“缓存 plan”生成下一段动作（可选 steps）
                    e_inputs = dict(base_inputs)
                    e_inputs["plan"] = current_plan or "保持编队并自保，必要时规避、占位。"
                    e_state = app.invoke_executor(e_inputs)  # 可能需要 ~5s
                    decision_result = e_state.get("result", "{}")

                    seq = parse_llm_decision_to_action_sequence(
                        decision_result,
                        default_hold_s=EXECUTOR_DEFAULT_HOLD_S,
                        max_steps=EXECUTOR_MAX_SEQUENCE_STEPS,
                        max_total_s=EXECUTOR_MAX_SEQUENCE_TOTAL_S,
                    )

                    # 将(hold_s, action)转为(end_time, action)
                    timeline: list[tuple[float, np.ndarray]] = []
                    t = sim_time
                    for hold_s, act in seq:
                        t = t + float(hold_s)
                        timeline.append((t, act.astype(np.float32, copy=False)))

                    with action_lock:
                        # 追加到末尾，避免覆盖还未执行完的动作
                        if action_timeline and action_timeline[-1][0] > sim_time:
                            action_timeline.extend(timeline)
                        else:
                            action_timeline = timeline
                        # 同步 last_action 为当前时刻应该执行的动作（更平滑）
                        if action_timeline:
                            last_action[:] = action_timeline[0][1]
                else:
                    final_state = app.invoke(base_inputs)  # 可能需要 ~5s
                    decision_result = final_state.get("result", "{}")
                    new_action = apply_llm_decision_to_sim(decision_result)
                    with action_lock:
                        last_action[:] = new_action
            except Exception as e:
                print(f"[LLM] 推理错误: {e}")
                # 出错时保持上一次动作/时间线，不做覆盖

    threading.Thread(target=llm_worker, daemon=True).start()

    try:
        while True:
            time_sim, sim_data = receiver.recv_frame(connection)

            if time_sim > reset_time_threshold:
                send_reset_instruction(connection)
                continue

            with action_lock:
                # 若存在动作序列，按 sim_time 推进
                while action_timeline and time_sim > action_timeline[0][0]:
                    action_timeline.pop(0)
                if action_timeline:
                    action_to_send = action_timeline[0][1].copy()
                    last_action[:] = action_to_send
                else:
                    action_to_send = last_action.copy()

            send_status_data(connection, action_to_send)
            push_latest_state((time_sim, sim_data))

    except KeyboardInterrupt:
        print("手动停止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        connection.close()
        server.close()
        print("Server closed.")

