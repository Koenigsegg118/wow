import argparse
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# --- 1. 定义数据模型 (兼容 OpenAI 格式) ---
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    thinking: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-custom"
    object: str = "chat.completion"
    created: int = 1234567890
    model: str
    choices: List[ChatCompletionResponseChoice]


# --- 2. 全局变量存储模型 ---
pipe = None
MODEL_NAME = ""

app = FastAPI()


# --- 3. 启动时的加载逻辑 ---
def load_model(model_path: str):
    global pipe, MODEL_NAME
    print(f"正在加载模型: {model_path} ... (这可能需要几分钟)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,  # 显存不够改 float16
            device_map="auto",
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True
        )
        MODEL_NAME = model_path
        print(f"模型 {model_path} 加载成功！")
    except Exception as e:
        print(f"!!! 模型加载失败: {e}")
        raise e


# --- 4. API 路由 ---

@app.get("/v1/models")
async def list_models():
    # 简单的 mock，让客户端确认连接成功
    return {
        "object": "list",
        "data": [{"id": "custom-model", "object": "model", "owned_by": "me"}]
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. 转换消息格式
    # 将 Pydantic 对象转为字典列表
    messages_list = [{"role": m.role, "content": m.content} for m in request.messages]

    # 2. 构建 Prompt
    try:
        prompt = pipe.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.thinking,
        )
    except Exception as e:
        # 如果 apply_chat_template 失败（某些旧模型），手动拼接
        print(f"Template apply warning: {e}")
        prompt = ""
        for m in messages_list:
            prompt += f"{m['role']}: {m['content']}\n"
        prompt += "assistant:"

    # 3. 推理
    outputs = pipe(
        prompt,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=request.temperature,
        top_p=request.top_p,
        return_full_text=False
    )

    generated_text = outputs[0]['generated_text']

    # 清理结束符
    if '<|im_end|>' in generated_text:
        generated_text = generated_text.split('<|im_end|>')[0].strip()

    # 4. 构造响应
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=generated_text),
                finish_reason="stop"
            )
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="本地模型路径")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    # 先加载模型
    load_model(args.model)

    # 再启动服务 (Windows 下 uvicorn 不会使用 uvloop)
    uvicorn.run(app, host="0.0.0.0", port=args.port)