import json
from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    task: str
    plan: Optional[str]
    result: Optional[str]
    dynamic_context: str
    planner_decision: Optional[str]


def call_vllm_model(
    client: Any,
    model_name: str,
    system_message: str,
    user_message: str,
    max_tokens: int = 1024,
) -> str:
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
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Error: {e}")
        return "{}"


def build_langgraph_app(
    *,
    planner_client: Any,
    executor_client: Any,
    planner_model_name: str,
    executor_model_name: str,
):
    try:
        from langgraph.graph import END, StateGraph  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'langgraph'. Please install it (see environment.yml) to build the graph."
        ) from e

    def planner_node(state: AgentState) -> dict:
        print("--- [Planner] ---")
        context = state["dynamic_context"]
        system_message = (
            "你是一个经验丰富的战斗机飞行员，能结合当前的状态作出合适的决策\n"
            "你需要对红方给出一个简明扼要的作战指令\n"
            "严格仅JSON 输出: {'action': 'execute', 'plan': '作战指令...'}"
        )
        user_message = f"[实时数据]: {context}\n请输出决策。"

        raw_output = call_vllm_model(
            planner_client, planner_model_name, system_message, user_message
        )
        print(f"Planner Output: {raw_output[:50]}...")  # 仅打印前50字符避免刷屏

        try:
            cleaned_output = raw_output.strip()
            if "```json" in cleaned_output:
                cleaned_output = (
                    cleaned_output.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_output:
                cleaned_output = cleaned_output.split("```")[1].strip()

            decision_json = json.loads(cleaned_output)
            plan = decision_json.get("plan", str(decision_json))
            return {"plan": plan, "planner_decision": "execute"}
        except Exception as e:
            print(f"Planner JSON Error: {e}")
            return {"plan": raw_output, "planner_decision": "execute"}

    def executor_node(state: AgentState) -> dict:
        print("--- [Executor] ---")
        plan = state["plan"]
        context_variable = state["dynamic_context"]

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
            "- dspeed_mps：相对当前速度的变化（单位：m/s，建议范围 [-50,50]）\n"
            "如果不需要控制某架飞机，把三个值都输出为 0。"
        )

        user_message = f"[指令]: {plan}\n[数据]: {context_variable}\n请生成信号。"

        result = call_vllm_model(
            executor_client, executor_model_name, system_message, user_message
        )
        print(f"Executor Result: {result}")
        return {"result": result}

    def router_node(state: AgentState) -> str:
        return "executor" if state["planner_decision"] == "execute" else END

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", router_node, {"executor": "executor", END: END})
    workflow.add_edge("executor", END)
    return workflow.compile()
