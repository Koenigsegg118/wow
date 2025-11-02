# ==== 0) 通用导入 ====
from typing import TypedDict, Literal, List, Dict

import threading
import torch
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


# ==== 1) 一个轻量的本地聊天封装（Transformers，本地推理；支持流式或一次性） ====
class LocalHFChat:
    def __init__(self, model_id: str, dtype=torch.float16, device_map="auto"):
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map
        )
        # 有些开源权重没pad，统一起来更稳
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token_id = self.tok.eos_token_id

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        # 兼容 Qwen/Llama 等常见 Instruct 模型的简洁角色模板
        sys = next((m["content"] for m in messages if m["role"] == "system"),
                   "你是专业、简洁的中文助手。")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        user = "\n\n".join(user_msgs) if user_msgs else ""
        return f"<|system|>\n{sys}\n<|user|>\n{user}\n<|assistant|>\n"

    def generate(self, messages: List[Dict[str, str]], *,
                 max_new_tokens=768, temperature=0.3, top_p=0.9, stream=False):
        prompt = self._build_prompt(messages)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            **inputs,
            do_sample=True,
            temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )

        # 明确分支:非流式立即返回 str
        if stream:
            # 流式分支
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            t = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            t.start()

            def stream_wrapper():
                for piece in streamer:
                    yield piece

            return stream_wrapper()  # 返回 generator
        else:
            # 非流式分支:直接返回完整文本
            with torch.no_grad():
                out = self.model.generate(**gen_kwargs)
            text = self.tok.decode(out[0], skip_special_tokens=True)
            # 提取 assistant 部分
            if "<|assistant|>" in text:
                return text.split("<|assistant|>")[-1].strip()
            return text.strip()


if __name__ == "__main__":
    # ==== 2) 加载两个“专家”模型（换成你的本地路径/仓库名） ====
    RESEARCHER_MODEL_PATH = "models/Qwen3-4B-Instruct-2507"
    CODER_MODEL_PATH = "models/gemma-3-1b-it"

    researcher_llm = LocalHFChat(RESEARCHER_MODEL_PATH, dtype=torch.bfloat16, device_map="auto")
    coder_llm = LocalHFChat(CODER_MODEL_PATH, dtype=torch.bfloat16, device_map="auto")

    # ==== 3) 定义全局状态（LangGraph 的“黑板”） ====
    class State(TypedDict, total=False):
        task: str  # 总任务
        messages: List[Dict[str, str]]  # 对话/日志（role: user/assistant/agent）
        result: str  # 最终产物（通常来自 coder）
        next_agent: Literal["researcher", "coder", "FINISH"]

    # ==== 4) 两个专家节点 ====
    def researcher(state: State) -> dict:
        """专家1：检索/调研/拆解"""
        q = state["task"]
        sys = "你是调研专家。目标：分解任务、列要点和注意事项；输出中文分点列表，务必具体。"
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": f"请针对任务给出要点：{q}"},
        ]
        # 关键：显式设置 stream=False，确保拿到的是 str
        draft = researcher_llm.generate(
            msgs, max_new_tokens=600, temperature=0.2, top_p=0.8, stream=False
        )
        log = {"role": "agent", "content": f"[researcher]\n{draft}"}
        return {
            "messages": state.get("messages", []) + [log],
            "next_agent": "coder",
        }


    def coder(state: State) -> dict:
        """专家2：把要点变成最终方案/代码"""
        notes = "\n".join(
            m["content"] for m in state.get("messages", [])
            if m["content"].startswith("[researcher]")
        )
        sys = "你是实现专家。把要点转为可执行方案或清晰的最终稿，结构化、可落地。"
        prompt = f"基于以下要点，输出最终方案（含步骤/伪代码/注意事项）：\n{notes or '(无)'}"
        msgs = [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ]
        # 同上：显式 stream=False
        final = coder_llm.generate(
            msgs, max_new_tokens=900, temperature=0.25, top_p=0.9, stream=False
        )
        log = {"role": "agent", "content": f"[coder]\n{final}"}
        return {
            "messages": state.get("messages", []) + [log],
            "result": final,  # 此时是 str，print 就不会是 generator 了
            "next_agent": "FINISH",
        }


    # ==== 5) 监督者节点（路由器） ====
    def supervisor(state: State) -> dict:
        """
        最小实现：规则路由（首次进来→researcher；之后看 next_agent）
        若将来想用“第三个本地模型”做智能路由，把这里改成 LLM 判定即可。
        """
        if "next_agent" not in state:
            return {"next_agent": "researcher"}
        return {}


    def route(state: State):
        nxt = state.get("next_agent", "researcher")
        return nxt if nxt in ("researcher", "coder") else "FINISH"


    # ==== 6) 组图并编译 ====
    builder = StateGraph(State)
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher)
    builder.add_node("coder", coder)
    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", route, {
        "researcher": "researcher",
        "coder": "coder",
        "FINISH": END
    })
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")
    graph = builder.compile()

    # ==== 7) 运行一次 ====
    init = {
        "task": "用Python抓取一个网页，清洗中文文本，统计出现频率Top10的词，并给出可复用的脚本结构。",
        "messages": []
    }
    out = graph.invoke(init, config={"recursion_limit": 10})
    print("\n=== 最终结果 ===\n", out["result"])
