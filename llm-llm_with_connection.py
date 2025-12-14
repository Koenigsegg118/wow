import time
import json
import operator
import socket
import struct
import requests
import numpy as np
from typing import TypedDict, Annotated, Optional
import requests
from requests.exceptions import RequestException
from openai import OpenAI
from langgraph.graph import StateGraph, END
import threading
from queue import Queue, Full

# ==========================================
# PART 1: LLM & LangGraph 配置 (保持不变)
# ==========================================

# --- 配置区域 ---
PLANNER_API_KEY = "EMPTY"
PLANNER_API_BASE = "http://10.134.114.3:5000/v1"
PLANNER_MODEL_NAME = "models/Qwen3-4B-Instruct-2507"

EXECUTOR_API_KEY = "EMPTY"
EXECUTOR_API_BASE = "http://10.134.114.3:5001/v1"
EXECUTOR_MODEL_NAME = "models/gemma3-1b-it"

# --- 初始化客户端 ---
def check_server_connection(base_url: str, server_name: str, timeout: int = 3):
    """
    测试服务器连接状态。
    尝试访问 /models 端点，这是 OpenAI 兼容接口的标准端点。
    """
    # print(f"正在测试 {server_name} 连接 ({base_url})...")
    try:
        # 去掉末尾可能的 /v1，通常健康检查可以直接访问根目录或 /v1/models
        # 这里我们严格测试 OpenAI 接口的连通性
        test_url = f"{base_url.rstrip('/')}/models"

        # 发送请求，设置超时时间防止程序卡死
        response = requests.get(test_url, timeout=timeout)

        if response.status_code == 200:
            print(f"✅ {server_name} 连接成功!")
        else:
            raise ConnectionError(f"服务器响应了，但状态码异常: {response.status_code}")

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"❌ 无法连接到 {server_name} ({base_url})。\n原因: 目标拒绝连接。请检查 IP 是否正确，端口是否开放，或防火墙设置。")
    except requests.exceptions.Timeout:
        raise ConnectionError(f"❌ 连接 {server_name} 超时 ({timeout}秒)。\n原因: 网络延迟过高或服务器无响应。")
    except Exception as e:
        raise ConnectionError(f"❌ 连接 {server_name} 发生未知错误: {e}")

print(f"正在连接规划者服务器: {PLANNER_API_BASE}...")
check_server_connection(PLANNER_API_BASE, "规划者 (Planner)")
planner_client = OpenAI(api_key=PLANNER_API_KEY, base_url=PLANNER_API_BASE)

print(f"正在连接执行者服务器: {EXECUTOR_API_BASE}...")
check_server_connection(EXECUTOR_API_BASE, "执行者 (Executor)")
executor_client = OpenAI(api_key=EXECUTOR_API_KEY, base_url=EXECUTOR_API_BASE)


class StateReceiver:
    """
    AFSIM发来的文本帧格式：
      simTime 640 v0 v1 ... v639
    由于TCP可能半包/粘包，这里用token缓冲，凑够一帧(2+state_len个token)再解析。
    """
    def __init__(self):
        self._text_buf = ""
        self._tokens = []

    def _feed(self, conn):
        data = conn.recv(8192)
        if not data:
            raise EOFError("socket closed")

        s = data.decode("ascii", errors="ignore")
        self._text_buf += s

        # 如果最后一个字符不是空白，说明最后一个token可能被截断，先留在buf里
        if self._text_buf and (not self._text_buf[-1].isspace()):
            parts = self._text_buf.split()
            if parts:
                self._text_buf = parts[-1]
                self._tokens.extend(parts[:-1])
            return

        # 结尾是空白 => token完整
        parts = self._text_buf.split()
        self._text_buf = ""
        self._tokens.extend(parts)

    def recv_frame(self, conn):
        while True:
            while len(self._tokens) < 2:
                self._feed(conn)

            sim_time = float(self._tokens[0])
            state_len = int(float(self._tokens[1]))
            need = 2 + state_len

            while len(self._tokens) < need:
                self._feed(conn)

            frame = self._tokens[:need]
            del self._tokens[:need]

            sim_time = float(frame[0])
            state_len = int(float(frame[1]))
            vals = np.asarray(frame[2:2 + state_len], dtype=np.float32)
            return sim_time, vals


# --- 状态定义 ---
class AgentState(TypedDict):
    task: str
    plan: Optional[str]
    result: Optional[str]
    dynamic_context: str
    planner_decision: Optional[str]


# --- 辅助函数 ---
def call_vllm_model(client: OpenAI, model_name: str, system_message: str, user_message: str,
                    max_tokens: int = 1024) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7, top_p=0.95, max_tokens=max_tokens, stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "{}"


# --- 节点定义 ---
def planner_node(state: AgentState) -> dict:
    # ... (代码与你之前提供的一致) ...
    print("--- [Planner] ---")
    context = state['dynamic_context']
    system_message = (
        "你是一个经验丰富的战斗机飞行员，能结合当前的状态作出合适的决策\n"
        "你需要对红方给出一个简明扼要的作战指令\n"
        "严格以 JSON 输出: {'action': 'execute', 'plan': '作战指令...'}"
    )
    user_message = f"[实时数据]: {context}\n请输出决策。"

    raw_output = call_vllm_model(planner_client, PLANNER_MODEL_NAME, system_message, user_message)
    print(f"Planner Output: {raw_output[:50]}...")  # 仅打印前50字符避免刷屏

    try:
        cleaned_output = raw_output.strip()
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].strip()

        decision_json = json.loads(cleaned_output)
        action = decision_json.get("action", "execute")  # 默认执行
        plan = decision_json.get("plan", str(decision_json))

        return {"plan": plan, "planner_decision": "execute"}
    except Exception as e:
        print(f"Planner JSON Error: {e}")
        return {"plan": raw_output, "planner_decision": "execute"}


def executor_node(state: AgentState) -> dict:
    # ... (代码与你之前提供的一致) ...
    print("--- [Executor] ---")
    plan = state['plan']
    context_variable = state['dynamic_context']

    system_message = (
        "你是一个飞机控制助手。根据指令与当前态势，只为红方两架飞机输出控制增量。\n"
        "必须严格输出 JSON（不要输出多余文字/代码块/解释）。\n"
        "JSON 格式固定为：\n"
        "{\n"
        '  "red_1": {"turn_deg": <float>, "up_m": <float>, "dspeed_mps": <float>},\n'
        '  "red_2": {"turn_deg": <float>, "up_m": <float>, "dspeed_mps": <float>}\n'
        "}\n"
        "含义：\n"
        "- turn_deg：相对当前航向的变化（单位：度，建议范围 [-45,45]）\n"
        "- up_m：相对当前高度的变化（单位：米，建议范围 [-2000,2000]）\n"
        "- dspeed_mps：相对当前速度的变化（单位：m/s，建议范围 [-150,150]）\n"
        "如果不需要控制某架飞机，把三个值都输出为 0。"
    )

    user_message = f"[指令]: {plan}\n[数据]: {context_variable}\n请生成信号。"

    result = call_vllm_model(executor_client, EXECUTOR_MODEL_NAME, system_message, user_message)
    print(f"Executor Result: {result}")
    return {"result": result}


def router_node(state: AgentState) -> str:
    return "executor" if state['planner_decision'] == 'execute' else END


# --- 构建图 ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", router_node, {"executor": "executor", END: END})
workflow.add_edge("executor", END)
app = workflow.compile()

# ==========================================
# PART 2: Socket 通信与数据转换 (核心修改)
# ==========================================

HOST = 'localhost'
PORT = 65432


# -----------------------------------------------------------
# [关键转换函数 1]: 将仿真数值数组转换为 LLM 可读的字符串
# -----------------------------------------------------------
def translate_sim_data_to_llm_context(time_sim, sim_data_np):
    """
    AFSIM(C++)发送结构为：simTime 640 (80个平台 * 8个值)
    每个平台 8 个值的含义：
      [0]=live, [1]=lat, [2]=lon, [3]=alt(m), [4]=velN, [5]=velE, [6]=velD, [7]=heading(deg)

    这里仅将前 4 个平台（0:red_1, 1:red_2, 2:blue_1, 3:blue_2）整理成 LLM 可读文本。
    """
    try:
        if sim_data_np is None or len(sim_data_np) < 8 * 4:
            return f"T={time_sim:.2f} (state too short: {len(sim_data_np) if sim_data_np is not None else 'None'})"

        def plat(idx: int):
            b = 8 * idx
            return {
                "live": int(sim_data_np[b + 0]),
                "lat": float(sim_data_np[b + 1]),
                "lon": float(sim_data_np[b + 2]),
                "alt_m": float(sim_data_np[b + 3]),
                "velN_mps": float(sim_data_np[b + 4]),
                "velE_mps": float(sim_data_np[b + 5]),
                "velD_mps": float(sim_data_np[b + 6]),
                "heading_deg": float(sim_data_np[b + 7]),
            }

        r1 = plat(0)
        r2 = plat(1)
        b1 = plat(2)
        b2 = plat(3)

        # 简明输出，避免上下文过长影响推理
        return (
            f"T={time_sim:.2f}\n"
            f"red_1: {r1}\n"
            f"red_2: {r2}\n"
            f"blue_1: {b1}\n"
            f"blue_2: {b2}\n"
        )
    except Exception as e:
        return f"Data Error: {str(e)}"


# -----------------------------------------------------------
# [关键转换函数 2]: 将 LLM 的 JSON 决策转换为控制信号
# -----------------------------------------------------------
def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def apply_llm_decision_to_sim(llm_result_str: str):
    """
    返回长度=640的 float32 动作数组：
    idx 0 -> red_1, idx 1 -> red_2, 每个平台 8 维里只用 0/1/2 三个:
      [0]=a0(dHeading_norm), [1]=a1(dAlt_norm), [2]=a2(dSpeed_norm)
    """
    action = np.zeros(640, dtype=np.float32)

    # 默认全 0，解析失败也不会乱动
    try:
        clean = llm_result_str.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].strip()

        cmd = json.loads(clean) if clean else {}
    except Exception:
        cmd = {}

    # 支持 red_1/red_2（推荐），也兼容你旧的 red1/red2
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


# # --- 通信辅助函数 (保持不变) ---
# def recv_proc(receive_str):
#     receive_str = receive_str.decode("ascii").strip()
#     parts = receive_str.split()
#     time_sim = int(float(parts[0]))
#     state_len = int(parts[1])
#     received_nums = np.array(parts[2:2 + state_len], dtype=np.float32)
#     return time_sim, received_nums


def send_status_data(connection, action_640_f32: np.ndarray):
    # 必须 640 float32
    action_640_f32 = np.asarray(action_640_f32, dtype=np.float32)
    if action_640_f32.size != 640:
        raise ValueError(f"action size must be 640 float32, got {action_640_f32.size}")

    connection.sendall(b"STATUS")
    # 小端 float32（Windows/Intel 默认就是小端；这里显式保证）
    payload = action_640_f32.astype("<f4", copy=False).tobytes()
    connection.sendall(payload)


def send_reset_instruction(connection):
    connection.sendall(b"RESET")


# ==========================================
# PART 3: 主循环 (Main Loop)
# ==========================================

if __name__ == '__main__':
    # 1. 启动服务器
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"Socket Server listening on {HOST}:{PORT}")
    print("等待仿真环境连接...")

    connection, address = server.accept()
    print(f"已连接: {address}")

    system_task = "作战模拟"

    # 禁用 Nagle，降低小包延迟（更容易在Afsim 0.1s窗口内收到 STATUS 前缀）
    try:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass

    # ------------------------------
    # 实时控制：立刻回包 + 后台慢推理
    # ------------------------------
    receiver = StateReceiver()

    # 永远保证有一个可用动作：初始全0（水平直飞）
    last_action = np.zeros(640, dtype=np.float32)
    action_lock = threading.Lock()

    # 只保留“最新状态”，避免LLM算5s导致队列堆积
    state_q: "Queue[tuple[float, np.ndarray]]" = Queue(maxsize=1)

    def push_latest_state(item):
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

    def llm_worker():
        while True:
            sim_time, sim_data = state_q.get()

            llm_context = translate_sim_data_to_llm_context(sim_time, sim_data)
            inputs = {"task": system_task, "dynamic_context": llm_context}

            try:
                final_state = app.invoke(inputs)   # 可能需要 ~5s
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
            # 1) 收到一整帧状态（带缓冲，防半包/粘包）
            time_sim, sim_data = receiver.recv_frame(connection)

            # 2) 立刻回包：把“上一帧动作/全0动作”发回去，保证AFSIM不超时
            if time_sim > 5000:
                send_reset_instruction(connection)
                continue

            with action_lock:
                action_to_send = last_action.copy()

            send_status_data(connection, action_to_send)

            # 3) 后台慢慢推理下一帧动作（只保留最新状态）
            push_latest_state((time_sim, sim_data))

            # 简短日志，避免刷屏
            # print(f"[T={time_sim:.2f}] sent immediate action; queued state for LLM.")

    except KeyboardInterrupt:
        print("手动停止")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        connection.close()
        server.close()
        print("Server closed.")