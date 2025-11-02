import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor, Gemma3ForConditionalGeneration,
    BitsAndBytesConfig
)
from trl import SFTTracondainer, SFTConfig
from peft import LoraConfig

model_id = "google/gemma-3-4b-it"

# 1) 量化 & 模型
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # 没装FA2会自动回退；能省显存就省
)
processor = AutoProcessor.from_pretrained(model_id)

# 2) LoRA 适配器
peft = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=[  # 对 Gemma 系列常用映射
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ]
)

# 3) 你的数据：需是 messages 结构
#   [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
# 这里用一个公开样例临时拼成 messages，实战请换成你自己的数据集。
raw = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train[:5000]")

def to_messages(ex):
    return {
        "messages": [
            {"role": "user", "content": f"把问题转成 SQL：\n{ex['sql_prompt']}\nSCHEMA:\n{ex['sql_context']}"},
            {"role": "assistant", "content": ex["sql"]}
        ]
    }

train_ds = raw.map(to_messages, remove_columns=raw.column_names)

# 4) 训练配置（24GB 友好）
args = SFTConfig(
    output_dir="./gemma3_4b_it_qlora_text",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=1000,
    bf16=True,
    gradient_checkpointing=True,   # 大幅省显存
    max_seq_length=4096,
    packing=True,                  # 高效拼样本
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    processing_class=processor,    # 让 TRL 自动套 chat 模板
    peft_config=peft,
)

trainer.train()
trainer.save_model()  # 保存 LoRA 适配器
processor.save_pretrained("./gemma3_4b_it_qlora_text")
