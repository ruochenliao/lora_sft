import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import BitsAndBytesConfig

###################### 量化配置 ###################################
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


model_name = 'Qwen/Qwen3-8B'
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=config,dtype=torch.bfloat16)
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

######################## 下面都是和 lora_sft 一致的  ####################
# 加载数据集
dataset_dict = load_dataset('json',data_files={"train":"keywords_data_train.jsonl","test":"keywords_data_test.jsonl"})
def map_func(example):
    conversation = example['conversation']
    messages = []
    for item in conversation:
        messages.append({'role':'user','content':item['human']})
        messages.append({'role':'assistant','content':item['assistant']})
    return {'messages':messages}

dataset_dict = dataset_dict.map(map_func,batched=False,remove_columns=['dataset','conversation','category','conversation_id'])


# ==================== PEFT LoRA 配置 ====================
# 参数高效微调 Parameter-Efficient Fine-Tuning

# TODO: Configure LoRA parameters
# 低秩矩阵的秩，越小压缩得越厉害，但是表达力越差；秩越大，与原始权重越接近，表达力约好
rank_dimension = 4
# 缩放因子，一般情况下是秩的2倍
lora_alpha = 8
# 防过拟合的 dropout
lora_dropout = 0.05
# 默认是none，训练它收益很小
bias = "none"

"""
注意力模块:
  q_proj  ← 线性层 
  k_proj  ← 线性层 
  v_proj  ← 线性层 
  o_proj  ← 线性层 

FFN 模块:
  gate_proj ← 线性层 
  up_proj   ← 线性层 
  down_proj ← 线性层 

"all-linear" 就是让 PEFT 自动找到模型中所有 nn.Linear 层，全部加上 LoRA。
"""
target_modules = "all-linear"

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)


# Configure trainer
training_args = SFTConfig(
    output_dir="./outputs/q_lora",
    max_steps=1000,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_steps=10,
    logging_dir='/root/tf-logs',
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    bf16=True,
    warmup_steps=50
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    processing_class=tokenizer,
    peft_config=peft_config
)

# 只会保存适配器
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))
print("input_ids_0=")
print(tokenizer.decode(batch['input_ids'][0]))


trainer.train()
trainer.save_model("./outputs/q_lora/best")
model_dtype = next(model.parameters()).dtype
print("model_dtype=")
print(model_dtype)



