from typing import Any


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.float16
)
tokenizer: Any = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model, "./outputs/best/", dtype=torch.float16
)

# 3. Merge adapter weights with base model
merged_model = peft_model.merge_and_unload()

# 4. save merged model
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")


# # 4. use merged model
# prompt = "抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别.该种类识别属于模式识别,其关键在于提取样品的特征参数,在获得特征参数的基础上,选取合适的网络通过训练来进行识别.猕猴桃种类识别的研究为自动化识别果品的种类、品种和新鲜等级等提供了一种新方法,为进一步研究果品介电特性与其内在品质的关系提供了一定的理论与实践基础."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
# )
# print(text)
# model_inputs = tokenizer([text], return_tensors="pt").to(merged_model.device)
# generated_ids = merged_model.generate(
#     **model_inputs,
#     max_new_tokens=32768
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# try:
#     # rindex finding 151668 (</think>)
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0

# thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# print("thinking content:", thinking_content)
# print("content:", content)