import socket
import threading
from queue import Full, Queue

import numpy as np

from .action_mapping import apply_llm_decision_to_sim
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
            inputs = {"task": system_task, "dynamic_context": llm_context}

            try:
                final_state = app.invoke(inputs)  # 可能需要 ~5s
                decision_result = final_state.get("result", "{}")
            except Exception as e:
                print(f"[LLM] 推理错误: {e}")
                decision_result = "{}"

            new_action = apply_llm_decision_to_sim(decision_result)
            with action_lock:
                last_action[:] = new_action

    threading.Thread(target=llm_worker, daemon=True).start()

    try:
        while True:
            time_sim, sim_data = receiver.recv_frame(connection)

            if time_sim > reset_time_threshold:
                send_reset_instruction(connection)
                continue

            with action_lock:
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

