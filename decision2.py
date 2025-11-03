import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator


# --- 1. 定义图的状态 (State) ---
# 状态是图在节点之间传递信息的方式。
class AgentState(TypedDict):
    task: str  # 用户的原始任务
    plan: str  # 规划者生成的计划
    result: str  # 执行者生成的最终结果
    planner_decision: str  # 规划者的决定: 'execute' 或 'end'
    messages: Annotated[list, operator.add]  # 用于多轮对话（可选，但推荐）


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
    """
    规划者节点：接收任务，生成规划。
    """
    print("--- 正在进入 [Planner] 节点 ---")
    task = state['task']

    system_message = (
        "你是一个高级AI规划师。"
        "你的目标是接收一个复杂的任务，并将其分解为一系列清晰、简洁、可执行的步骤。"
        "不要执行这些步骤，只输出步骤列表。"
    )
    # system_message = (
    #     "你是一个高级AI规划师。\n"
    #     "你的目标是接收一个任务，并决定是需要一个详细的执行计划，还是可以直接回答。\n"
    #     "你必须以 JSON 格式输出你的决定。\n\n"
    #     "1. 如果任务复杂，需要执行者处理，请输出: \n"
    #     '{"action": "execute", "plan": "这里是你的详细步骤..."}\n\n'
    #     "2. 如果任务是一个简单的问题（例如 '今天天气如何' 或 '2+2等于几'），你可以直接回答，请输出: \n"
    #     '{"action": "end", "answer": "这里是你的直接答案..."}\n\n'
    #     "确保你的输出是且仅是一个有效的JSON。"
    # )

    raw_plan = call_local_model(planner_pipe, system_message, task)

    print(f"规划者生成的计划:\n{raw_plan}")

    return {"plan": raw_plan, "messages": [("planner", raw_plan)]}

    try:
        # 尝试解析模型的JSON输出
        # 有时模型会在JSON前后添加```json ... ```, 我们尝试清理
        if "```json" in raw_plan:
            raw_plan = raw_plan.split("```json")[1].split("```")[0].strip()

        decision_json = json.loads(raw_plan)
        action = decision_json.get("action")

        if action == "execute":
            plan = decision_json.get("plan", "No plan provided")
            print("规划者决定: [执行]")
            return {
                "plan": plan,
                "planner_decision": "execute",
                "messages": [("planner", raw_plan)]
            }

        elif action == "end":
            answer = decision_json.get("answer", "No answer provided")
            print("规划者决定: [结束]")
            # 如果规划者直接回答，我们将此答案放入 'result' 字段
            return {
                "result": answer,
                "planner_decision": "end",
                "messages": [("planner", raw_plan)]
            }

        else:
            # 如果JSON无效或缺少 'action'
            raise ValueError("JSON中缺少 'action' 字段或 'action' 值无效")

    except json.JSONDecodeError as e:
        print(f"!!! 错误: 规划者未能输出有效的JSON。错误: {e} !!!")
        print(f"原始输出: {raw_plan}")
        # 错误处理：默认强制执行
        print("默认操作: 强制 [执行]")
        return {
            "plan": f"JSON解析失败。原始输出: {raw_plan}",
            "planner_decision": "execute",
            "messages": [("planner", f"JSON Error: {raw_plan}")]
        }



def executor_node(state: AgentState) -> dict:
    """
    执行者节点：接收规划，执行任务，生成最终结果。
    """
    print("--- 正在进入 [Executor] 节点 ---")
    task = state['task']
    plan = state['plan']

    system_message = (
        "你是一个AI执行助手。"
        "你的目标是严格遵循用户提供的“计划”，并根据“原始任务”生成详细的最终答案。"
        "请确保你的回答完整并直接解决了原始任务。"
    )

    # 将任务和计划都提供给执行者
    user_message = f"""
    原始任务: {task}

    请遵循以下计划来完成任务:
    {plan}

    请立即开始执行并提供最终结果。
    """

    result = call_local_model(executor_pipe, system_message, user_message)

    print(f"执行者生成的结果:\n{result}")

    return {"result": result, "messages": [("executor", result)]}


# --- 4. 定义条件路由函数 (Router) ---
def router_node(state: AgentState) -> str:
    """
    决策节点：检查规划者的决定，并返回下一个节点的名称。
    """
    print("--- 正在进入 [Router] 决策点 ---")

    if state['planner_decision'] == 'execute':
        print("路由决策: 前往 [Executor]")
        return "executor"  # 返回节点名称字符串
    else:
        print("路由决策: [结束]")
        return END  # 返回特殊的 END 标记


# --- 5. 构建图 (Graph) ---

# 初始化 StateGraph
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

# 设置入口点
workflow.set_entry_point("planner")

# 添加边
# 规划者 -> 执行者
workflow.add_edge("planner", "executor")

# # 从 "planner" 节点开始，调用 router_node 函数
# # router_node 函数的返回值（"executor" 或 END）将决定下一步去哪里
# workflow.add_conditional_edges(
#     "planner",  # 起始节点
#     router_node, # 调用的路由函数
#     {
#         "executor": "executor", # 如果路由函数返回 "executor", 则转到 "executor" 节点
#         END: END                # 如果路由函数返回 END, 则结束
#     }
# )

# 执行者是最后一个节点
workflow.add_edge("executor", END)

# 编译图
app = workflow.compile()

# --- 5. 运行图 ---

if __name__ == "__main__":
    system_task = "我应该如何为即将到来的技术面试做准备？请给我一个关于数据结构和算法的学习方向"

    # LangGraph 支持流式 (stream) 和 批处理 (invoke)
    # 我们使用 invoke 来获取最终结果

    inputs = {"task": system_task, "messages": [("user", system_task)]}

    final_state = app.invoke(inputs)

    print("\n--- 最终结果 ---")
    print(final_state['result'])

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