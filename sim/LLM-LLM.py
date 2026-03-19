import time
import random
import json
import operator
from typing import TypedDict, Annotated, Optional
from openai import OpenAI  # 使用 OpenAI 客户端连接 vLLM
from langgraph.graph import StateGraph, END

# ==========================================
# 0. 配置区域 (请根据您的 vLLM 服务器修改)
# ==========================================

# --- 规划者 (Planner) 配置 ---
# 假设规划者模型部署在 8000 端口
PLANNER_API_KEY = "EMPTY"  # vLLM 通常不需要 key，填 EMPTY 即可
PLANNER_API_BASE = "http://10.134.114.3:5000/v1"  # 替换为您服务器的 IP，例如 http://192.168.1.10:8000/v1
PLANNER_MODEL_NAME = "models/Qwen3-4B-Instruct-2507"  # 必须与 vLLM 启动时的 --model 参数一致

# --- 执行者 (Executor) 配置 ---
# 假设执行者模型部署在 8001 端口 (如果是同一个服务，只是模型不同，改端口或模型名即可)
EXECUTOR_API_KEY = "EMPTY"
EXECUTOR_API_BASE = "http://10.134.114.3:5001/v1"  # 替换为您服务器的 IP
EXECUTOR_MODEL_NAME = "models/gemma-3-1b-it"

# 初始化客户端
print(f"正在连接规划者服务器: {PLANNER_API_BASE}...")
planner_client = OpenAI(api_key=PLANNER_API_KEY, base_url=PLANNER_API_BASE)

print(f"正在连接执行者服务器: {EXECUTOR_API_BASE}...")
executor_client = OpenAI(api_key=EXECUTOR_API_KEY, base_url=EXECUTOR_API_BASE)


# ==========================================
# 1. 定义图的状态 (State)
# ==========================================
class AgentState(TypedDict):
    task: str  # 用户的原始任务
    plan: Optional[str]  # 规划者生成的计划
    result: Optional[str]  # 执行者生成的最终结果
    dynamic_context: str  # 实时数据
    planner_decision: Optional[str]


# ==========================================
# 2. 辅助函数：调用 vLLM API
# ==========================================
def call_vllm_model(client: OpenAI, model_name: str, system_message: str, user_message: str,
                    max_tokens: int = 1024) -> str:
    """
    通过 OpenAI 兼容接口调用 vLLM 部署的模型
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_tokens,
            stream=False  # 这里使用非流式，如果需要流式需额外处理
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用错误 (Model: {model_name}): {e}")
        # 返回空字符串或错误提示，防止程序崩溃
        return "{}"

    # ==========================================


# 3. 定义图的节点 (Nodes)
# ==========================================

def planner_node(state: AgentState) -> dict:
    print("--- 正在进入 [Planner] 节点 ---")
    # task = state['task'] # 这里的 task 暂时没用到，主要看 dynamic_context
    context = state['dynamic_context']

    system_message = (
        "你是一个经验丰富的战斗机飞行员，能结合当前的状态作出合适的决策\n"
        "现在我将给你红方和蓝方相对固定坐标系的速度、位置信息，你需要对红方给出一个简明扼要的作战指令，而不是直接输出飞机的运动学控制信息"
        "你必须严格以 JSON 格式输出你的决定。\n"
        '{"action": "execute", "plan": "作战指令..."}\n'
    )

    user_message = f"""
    [实时数据]: {context}

    请根据上述数据输出你的JSON决策。
    """

    # 调用 vLLM
    raw_output = call_vllm_model(
        planner_client,
        PLANNER_MODEL_NAME,
        system_message,
        user_message,
        max_tokens=1024
    )

    print(f"规划者 (原始JSON输出):\n{raw_output}")

    try:
        # 清理可能的 markdown 代码块标记 (vLLM 生成的代码经常包含 ```json)
        cleaned_output = raw_output.strip()
        if "```json" in cleaned_output:
            cleaned_output = cleaned_output.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_output:
            cleaned_output = cleaned_output.split("```")[1].strip()

        decision_json = json.loads(cleaned_output)
        action = decision_json.get("action")

        if action == "execute":
            return {"plan": decision_json.get("plan"), "planner_decision": "execute"}
        elif action == "end":
            return {"result": decision_json.get("answer"), "planner_decision": "end"}
        else:
            # 如果没有 action 字段，尝试宽容处理
            if "plan" in decision_json:
                return {"plan": decision_json.get("plan"), "planner_decision": "execute"}
            raise ValueError("Invalid action or missing plan")

    except Exception as e:
        print(f"JSON解析错误: {e}")
        # 错误回退：强制执行，将原始输出作为计划
        return {"plan": f"JSON解析失败，原始数据: {raw_output}", "planner_decision": "execute"}


def executor_node(state: AgentState) -> dict:
    print("--- 正在进入 [Executor] 节点 ---")
    plan = state['plan']
    context_variable = state['dynamic_context']

    system_message = (
        "你是一个飞机控制助手\n"
        "你需要根据[作战指令]和[实时数据]生成具体的控制飞机信号\n"
        "你的指令格式是“红方对象”+“动作”+“运动量”，其中红方对象为red1和red2；动作分为turn、updown，分别表示转弯、爬升\下降；当转弯时，运动量为转弯的角度，转弯角度有30 60可选，向左转为正，向右为负，爬升\下降时，运动量为对应z轴位移量，爬升距离仅100可选，上升为正、下降为负。\n"
        "请严格按照以下json示例生成控制信号\n"
        "{'red1': 'turn -30', 'red2': 'up 100'}"
        # "##red1 turn -30 red2 up 100##\n"
    )

    user_message = f"""
        [作战指令]: {plan}
        [实时数据]: {context_variable}

        请生成最终的决策报告。
        """

    # 调用 vLLM
    result = call_vllm_model(
        executor_client,
        EXECUTOR_MODEL_NAME,
        system_message,
        user_message,
        max_tokens=2048
    )

    print(f"执行者生成的结果:\n{result}")
    return {"result": result}


def router_node(state: AgentState) -> str:
    # print("--- 正在进入 [Router] 决策点 ---")
    if state['planner_decision'] == 'execute':
        return "executor"
    else:
        return END


# ==========================================
# 4. 构建图 (Graph)
# ==========================================

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    router_node,
    {"executor": "executor", END: END}
)

workflow.add_edge("executor", END)

app = workflow.compile()

# ==========================================
# 5. 运行图 (主循环)
# ==========================================
if __name__ == "__main__":
    system_task = "作战模拟"
    SLEEP_INTERVAL_SECONDS = 0.5

    print(f"--- 启动 vLLM 驱动的自主决策系统 ---")
    print(f"运行间隔: {SLEEP_INTERVAL_SECONDS} 秒")

    cnt = 0

    # 模拟数据 (这里您可以替换为真实的 API 调用)
    # 为了测试效果，我们稍微改变一下输入数据，看看反应
    mock_context_data = (
        "红方:\n"
        "{'state': {'heading': 0, 'position': (5000.0, 20000.0, 500), 'speed': 200}}\n"
        "{'state': {'heading': 0, 'position': (5000.0, 35000.0, 500), 'speed': 200}}\n"
        "蓝方:\n"
        "{'state': {'heading': 180, 'position': (45000.0, 25000.0, 500), 'speed': 200}}\n"
        "{'state': {'heading': 180, 'position': (45000.0, 30000.0, 500), 'speed': 200}}\n"
    )

    while True:
        cnt += 1
        try:
            start_time = time.time()

            inputs = {
                "task": system_task,
                "dynamic_context": mock_context_data
            }

            # 调用图
            final_state = app.invoke(inputs)

            # 输出决策
            decision = final_state.get('result', '没有生成决策')
            end_time = time.time()

            print(f"[{time.ctime()}] --- 系统决策 (耗时 {end_time - start_time:.2f}s) ---")
            print(decision)
            print("-" * 20)

        except Exception as e:
            print(f"[{time.ctime()}] [严重错误] 决策循环出错: {e}")

        # 简单控制循环次数用于测试，实际部署可去掉
        if cnt >= 5:
            print("测试结束。")
            break

        time.sleep(SLEEP_INTERVAL_SECONDS)