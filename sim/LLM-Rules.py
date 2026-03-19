# 这里下层改用基于规则的控制器，只是比较简单的规则

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, END
from typing import TypedDict
import time
import random
import json
import math
import re
import ast


# --- 1. 定义图的状态 (State) ---
class AgentState(TypedDict):
    task: str
    plan: str | None
    result: str | None
    dynamic_context: str
    planner_decision: str | None


# --- 2. 加载本地模型 (仅规划者 Planner) ---
# 注意：这里我们只加载一个负责思考的大脑
planner_model_path = "models/Qwen3-4B-Instruct-2507"  # 请确保路径正确
print(f"正在加载规划者模型 (Brain): {planner_model_path}")

try:
    planner_tokenizer = AutoTokenizer.from_pretrained(planner_model_path)
    planner_model = AutoModelForCausalLM.from_pretrained(
        planner_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    planner_pipe = pipeline(
        "text-generation",
        model=planner_model,
        tokenizer=planner_tokenizer,
        max_new_tokens=512,  # 规划不需要太长
        do_sample=True,
        temperature=0.6,  # 稍微降低温度，让决策更稳定
        top_p=0.9
    )
    print("规划者模型加载完毕。")
except Exception as e:
    print(f"模型加载失败，请检查路径: {e}")
    # 为了演示代码逻辑，如果没有模型，这里可以是一个Mock对象，实际运行时请注释掉
    planner_pipe = None


# --- 辅助函数：调用 LLM ---
def call_local_model(pipe, system_message, user_message):
    if pipe is None:
        # Mock 返回，用于调试代码逻辑
        return json.dumps({"action": "execute", "plan": "intercept"})

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, return_full_text=False)
    generated_text = outputs[0]['generated_text']
    if '<|im_end|>' in generated_text:
        generated_text = generated_text.split('<|im_end|>')[0].strip()
    return generated_text


# --- 3. 新增：传统算法执行层 (The "Math" Layer) ---

class BVRRuleBasedExecutor:
    """
    这是一个不使用 LLM 的执行器。
    它接收战术意图（Plan）和数据（Context），通过几何计算生成控制信号。
    """

    def parse_context_string(self, context_str):
        """解析有点混乱的字符串数据为结构化字典"""
        units = {'red': [], 'blue': []}

        # 使用简单的正则提取字典部分
        # 假设每一行是一个单位的状态
        lines = context_str.strip().split('\n')
        current_side = None

        for line in lines:
            if '红方' in line:
                current_side = 'red'
                continue
            elif '蓝方' in line:
                current_side = 'blue'
                continue

            # 提取类似 {'state': ...} 的字符串
            match = re.search(r"\{.*\}", line)
            if match and current_side:
                try:
                    # 将字符串转为字典
                    data = ast.literal_eval(match.group(0))
                    # 为单位添加一个临时ID
                    unit_id = f"{current_side}{len(units[current_side]) + 1}"
                    data['id'] = unit_id
                    units[current_side].append(data)
                except:
                    pass
        return units

    def calculate_bearing(self, p1, p2):
        """计算从 p1 到 p2 的方位角 (0-360)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # math.atan2 返回的是弧度，X轴正向为0，逆时针为正
        # 这里的坐标系通常 北为Y正，东为X正。
        # atan2(dy, dx) -> 0度是正东，90度是正北
        angle = math.degrees(math.atan2(dx, dy))  # 注意：这里为了对应航向角(0=北, 90=东)，参数顺序可能需要根据你的坐标系调整

        # 归一化到 0-360
        if angle < 0:
            angle += 360
        return angle

    def get_turn_command(self, current_heading, target_heading):
        """根据当前航向和目标航向，决定 turn 30 还是 -30"""
        diff = target_heading - current_heading

        # 处理跨越 360/0 度的情况
        if diff > 180: diff -= 360
        if diff < -180: diff += 360

        # 死区，小于5度不转
        if abs(diff) < 5:
            return ""

            # 简单的 P 控制逻辑 (Proportional)
        # 你的接口只支持 30 或 60
        direction = 1 if diff > 0 else -1  # 1 is right (check logic), -1 is left

        # 假设：向右转为正(turn 30)，向左转为负(turn -30)
        # 注意：你需要根据你的仿真环境确认正负号定义
        # 这里假设：Heading 增加是向右转

        abs_diff = abs(diff)
        angle_mag = 60 if abs_diff > 45 else 30

        final_angle = angle_mag * direction
        return f"turn {final_angle}"

    def execute(self, plan_text, context_str):
        """核心执行逻辑"""
        data = self.parse_context_string(context_str)
        commands = []

        # 简单的战术状态机
        # 1. 识别意图 (从 LLM 的 Plan 中提取关键词)
        intent = "intercept"  # 默认
        if "retreat" in plan_text.lower() or "撤退" in plan_text:
            intent = "retreat"
        elif "climb" in plan_text.lower() or "爬升" in plan_text:
            intent = "climb"

        # 2. 为每个红方单位计算指令
        for red in data['red']:
            cmd_parts = [red['id']]  # e.g. "red1"

            # --- 逻辑 A: 导航/机动逻辑 ---
            if intent == "intercept":
                # 寻找最近的蓝军
                target = data['blue'][0] if data['blue'] else None
                if target:
                    pos_r = red['state']['position']
                    pos_b = target['state']['position']

                    # 几何计算：这就体现了传统算法的优势，精确！
                    target_bearing = self.calculate_bearing(pos_r, pos_b)
                    current_heading = red['state']['heading']

                    turn_cmd = self.get_turn_command(current_heading, target_bearing)
                    if turn_cmd:
                        cmd_parts.append(turn_cmd)

            elif intent == "retreat":
                cmd_parts.append("turn 60")  # 紧急脱离

            # --- 逻辑 B: 高度控制 ---
            # 简单的规则：如果意图是爬升，或者太低了
            z_alt = red['state']['position'][2]
            if intent == "climb" or z_alt < 1000:
                cmd_parts.append("up 100")
            elif z_alt > 8000:  # 太高了就下来
                cmd_parts.append("up -100")

            # 组合指令
            if len(cmd_parts) > 1:  # 只有当有动作时才添加
                commands.append(" ".join(cmd_parts))

        # 3. 格式化最终输出
        return "##" + " ".join(commands) + "##"


# 实例化算法执行器
rule_based_executor = BVRRuleBasedExecutor()


# --- 4. 定义图的节点 (Nodes) ---

def planner_node(state: AgentState) -> dict:
    print("--- 正在进入 [Planner - LLM Brain] 节点 ---")
    context = state['dynamic_context']

    # Prompt Engineering:
    # 指导 LLM 输出 High-Level 的战术关键词，而不是底层控制参数
    system_message = (
        "你是一名红方空战指挥官。请根据实时坐标数据，分析态势并下达战术意图。\n"
        "你的输出必须是严格的 JSON 格式。\n"
        "可用的战术意图(keywords)：'intercept' (拦截/进攻), 'retreat' (撤退/防御), 'climb' (爬升/巡逻)\n"
        "示例: {\"action\": \"execute\", \"plan\": \"intercept\", \"reason\": \"敌机距离接近，且我方有能量优势\"}"
    )

    user_message = f"[实时数据]:\n{context}\n\n请输出决策 JSON:"

    raw_output = call_local_model(planner_pipe, system_message, user_message)
    print(f"规划者输出 (意图): {raw_output}")

    try:
        decision_json = json.loads(raw_output)
        action = decision_json.get("action", "execute")
        # 提取 LLM 的战术意图，传递给算法层
        plan_content = decision_json.get("plan", "intercept")

        return {"plan": plan_content, "planner_decision": action}

    except Exception as e:
        print(f"JSON解析错误: {e}")
        return {"plan": "intercept", "planner_decision": "execute"}  # 默认兜底策略


def executor_node(state: AgentState) -> dict:
    """
    执行者节点：现在这里是【纯代码逻辑】，没有 LLM 了！
    """
    print("--- 正在进入 [Executor - Algorithm Control] 节点 ---")

    plan_intent = state['plan']  # 来自 LLM 的意图 (如 "intercept")
    context = state['dynamic_context']  # 原始数据

    # 调用我们的数学算法类
    # 这步计算是毫秒级的，且绝对精准，不会出现幻觉
    result = rule_based_executor.execute(plan_intent, context)

    print(f"执行者计算结果 (Control Signals):\n{result}")

    return {"result": result}


def router_node(state: AgentState) -> str:
    if state['planner_decision'] == 'execute':
        return "executor"
    else:
        return END


# --- 5. 构建图 (Graph) ---

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)  # 这里的 executor 已经是算法版了
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", router_node, {"executor": "executor", END: END})
workflow.add_edge("executor", END)
app = workflow.compile()

# --- 6. 运行仿真 ---
if __name__ == "__main__":
    system_task = "BVR Combat Sim"

    # 模拟两帧数据，展示算法如何根据几何位置自动调整转向
    simulated_contexts = [
        # Frame 1: 蓝机在右前方，算法应该计算出右转 (turn 30 或 60)
        ("红方:\n{'state': {'heading': 0, 'position': (0.0, 0.0, 5000), 'speed': 300}}\n"
         "蓝方:\n{'state': {'heading': 180, 'position': (5000.0, 5000.0, 5000), 'speed': 300}}\n"),

        # Frame 2: 蓝机在左前方，算法应该计算出左转
        ("红方:\n{'state': {'heading': 0, 'position': (0.0, 0.0, 5000), 'speed': 300}}\n"
         "蓝方:\n{'state': {'heading': 180, 'position': (-5000.0, 5000.0, 5000), 'speed': 300}}\n")
    ]

    for i, ctx in enumerate(simulated_contexts):
        print(f"\n====== 仿真帧 {i + 1} ======")
        inputs = {
            "task": system_task,
            "dynamic_context": ctx
        }

        final_state = app.invoke(inputs)

        print("\n[最终控制信号]")
        print(final_state['result'])
        print("-" * 30)