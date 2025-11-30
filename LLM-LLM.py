# 这里是上下层均使用llm的版本，opt微调可能会在这个系统中使用

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import time
import random
import json


# --- 1. 定义图的状态 (State) ---
# 状态是图在节点之间传递信息的方式。
class AgentState(TypedDict):
    task: str  # 用户的原始任务
    plan: str | None  # 规划者生成的计划
    result: str | None  # 执行者生成的最终结果
    dynamic_context: str
    planner_decision: str | None


# --- 2. 加载本地模型 (使用 Transformers) ---
# 您需要将这里的 model_id 替换为您本地模型的路径或Hugging Face ID
# 为了演示，我们加载两个“不同”的模型实例
# 在实际应用中，您可以为规划者和执行者加载完全不同的模型

# 规划者模型
planner_model_path = "models/Qwen3-4B-Instruct-2507"
print(f"正在加载规划者模型: {planner_model_path}")
planner_tokenizer = AutoTokenizer.from_pretrained(planner_model_path)
planner_model = AutoModelForCausalLM.from_pretrained(
    planner_model_path,
    dtype=torch.bfloat16,  # 使用 float16 节省显存
    device_map="auto"  # 自动分配到 GPU (如果可用)
)
# 使用 Transformers Pipeline 封装
planner_pipe = pipeline(
    "text-generation",
    model=planner_model,
    tokenizer=planner_tokenizer,
    max_new_tokens=1024,  # 限制生成长度
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

# 执行者模型
# 为简单起见，我们使用同一个模型，但您可以加载另一个
# executor_model_id = "path/to/your/executor/model"
executor_model_path = "models/gemma-3-1b-it"

print(f"正在加载执行者模型: {executor_model_path}")
executor_tokenizer = AutoTokenizer.from_pretrained(executor_model_path)
executor_model = AutoModelForCausalLM.from_pretrained(
    executor_model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)
executor_pipe = pipeline(
    "text-generation",
    model=executor_model,
    tokenizer=executor_tokenizer,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

print("模型加载完毕。")


# --- 辅助函数：调用本地模型 ---
# 这是一个帮助函数，用于格式化prompt并调用pipeline
def call_local_model(pipe: pipeline, system_message: str, user_message: str) -> str:
    """
    使用 transformers pipeline 调用本地模型。
    使用 Qwen2 的聊天模板。
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # apply_chat_template 会自动处理特殊的 tokens (如 <|im_start|>)
    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Pipeline 会返回一个列表
    outputs = pipe(prompt, return_full_text=False)
    generated_text = outputs[0]['generated_text']

    # 清理可能的tokenizer残留
    if '<|im_end|>' in generated_text:
        generated_text = generated_text.split('<|im_end|>')[0].strip()

    return generated_text


# --- 3. 定义图的节点 (Nodes) ---

def planner_node(state: AgentState) -> dict:
    print("--- 正在进入 [Planner] 节点 ---")
    task = state['task']
    context = state['dynamic_context']  # <--- 获取实时数据

    # system_message = (
    #     "你是一个自主决策AI。你的目标是分析实时数据，并决定是否需要详细执行，还是可以直接给出决策。\n"
    #     "你必须以 JSON 格式输出你的决定。\n"
    #     '1. (复杂决策): {"action": "execute", "plan": "执行步骤..."}\n'
    #     '2. (简单决策): {"action": "end", "answer": "决策结果..."}\n'
    #     "现在测试功能，请只输出复杂决策"
    # )

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

    raw_output = call_local_model(planner_pipe, system_message, user_message)
    print(f"规划者 (原始JSON输出):\n{raw_output}")

    try:
        # ... (JSON 解析逻辑，与之前相同) ...
        # (假设解析出了 'action', 'plan', 'answer')

        decision_json = json.loads(raw_output)  # (简化)
        action = decision_json.get("action")

        if action == "execute":
            return {"plan": decision_json.get("plan"), "planner_decision": "execute"}
        elif action == "end":
            return {"result": decision_json.get("answer"), "planner_decision": "end"}
        else:
            raise ValueError("Invalid action")

    except Exception as e:
        print(f"JSON解析错误: {e}")
        # 错误回退：强制执行
        return {"plan": f"JSON解析失败，原始数据: {raw_output}", "planner_decision": "execute"}



def executor_node(state: AgentState) -> dict:
    """
    执行者节点：接收规划，执行任务，生成最终结果。
    """
    print("--- 正在进入 [Executor] 节点 ---")
    # task = state['task']
    plan = state['plan']

    context_variable = state['dynamic_context']

    # system_message = (
    #     "你是一个AI执行助手。"
    #     "严格遵循“计划”，并结合“实时数据”来生成简短决策报告。"
    # )

    system_message = (
        "你是一个飞机控制助手\n"
        "你需要根据[作战指令]和[实时数据]生成具体的控制飞机信号\n"
        "你的指令格式是“红方对象”+“动作”+“运动量”，其中红方对象为red1和red2；动作分为turn、updown，分别表示转弯、爬升\下降；当转弯时，运动量为转弯的角度，转弯角度有30 60可选，向左转为正，向右为负，爬升\下降时，运动量为对应z轴位移量，爬升距离仅100可选，上升为正、下降为负。\n"
        "请严格按照以下示例生成控制信号\n"
        "##red1 turn -30 red2 up 100##\n"
    )

    user_message = f"""
        [作战指令]: {plan}
        [实时数据]: {context_variable}

        请生成最终的决策报告。
        """

    result = call_local_model(executor_pipe, system_message, user_message)

    print(f"执行者生成的结果:\n{result}")

    return {"result": result, "messages": [("executor", result)]}

def router_node(state: AgentState) -> str:
    print("--- 正在进入 [Router] 决策点 ---")
    if state['planner_decision'] == 'execute':
        return "executor"
    else:
        return END

# --- 5. 构建图 (Graph) ---

# 初始化 StateGraph
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

# 设置入口点
workflow.set_entry_point("planner")


# # 从 "planner" 节点开始，调用 router_node 函数
# router_node 函数的返回值（"executor" 或 END）将决定下一步去哪里


workflow.add_conditional_edges(
    "planner",
    router_node,
    {"executor": "executor", END: END}
)

# 执行者是最后一个节点
workflow.add_edge("executor", END)

# 编译图
app = workflow.compile()

# --- 模拟的 API 函数 --- 会被仿真环境api替代
def get_realtime_data_from_api() -> str:
    """
    模拟一个API调用，它会返回一个动态变化的值。
    """
    print(f"\n[{time.ctime()}] ... 正在调用实时 API ...")
    states = [
        "市场状态：看涨 (Bullish)",
        "市场状态：看跌 (Bearish)",
        "市场状态：横盘 (Sideways)"
    ]
    time.sleep(0.5) # 模拟网络延迟
    data = random.choice(states)
    print(f"[{time.ctime()}] ... API 返回: {data} ...")
    return data

# --- 5. 运行图 ---
if __name__ == "__main__":
    system_task = "作战模拟"

    SLEEP_INTERVAL_SECONDS = 0.5

    print(f"--- 启动自主决策系统 ---")
    # print(f"固定任务: {system_task}")
    print(f"运行间隔: {SLEEP_INTERVAL_SECONDS} 秒")
    cnt = 0
    while True:
        cnt += 1
        try:
            # api_data = get_realtime_data_from_api()
            inputs = {
                "task": system_task,
                # "dynamic_context": api_data
                "dynamic_context": "红方:\n"
            "{'state': {'heading': 0, 'position': (5000.0, 20000.0, 500), 'speed': 200}}\n"
            "{'state': {'heading': 0, 'position': (5000.0, 35000.0, 500), 'speed': 200}}\n"
            "蓝方:\n"
            "{'state': {'heading': 180, 'position': (45000.0, 25000.0, 500), 'speed': 200}}\n"
            "{'state': {'heading': 180, 'position': (45000.0, 30000.0, 500), 'speed': 200}}\n"
            }

            final_state = app.invoke(inputs)

            # 4. 输出决策
            decision = final_state.get('result', '没有生成决策')
            print(f"[{time.ctime()}] --- 系统决策 ---")
            print(decision)
            print("-" * 20)

        except Exception as e:
            print(f"[{time.ctime()}] [严重错误] 决策循环出错: {e}")
            # 即使出错，也继续循环

            # 5. 等待下一个周期
        time.sleep(SLEEP_INTERVAL_SECONDS)

        print("\n--- 最终结果 ---")
        print(final_state['result'])
        if cnt == 5:
            break

    # print("======= 测试 1: 复杂任务 (应调用 Executor) =======")
    # task_complex = "我应该如何为即将到来的技术面试做准备？请给我一个关于数据结构和算法的学习计划。"
    # inputs_complex = {"task": task_complex, "messages": [("user", task_complex)]}
    #
    # final_state_complex = app.invoke(inputs_complex)
    #
    # print("\n--- 复杂任务的最终结果 ---")
    # print(final_state_complex['result'])
    #
    # print("\n" + "=" * 30 + "\n")
    #
    # print("======= 测试 2: 简单任务 (应直接结束) =======")
    # task_simple = "今天星期几？"
    # inputs_simple = {"task": task_simple, "messages": [("user", task_simple)]}
    #
    # final_state_simple = app.invoke(inputs_simple)
    #
    # print("\n--- 简单任务的最终结果 ---")
    # print(final_state_simple['result'])