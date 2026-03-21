from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Downloading Qwen 2.5...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.float16
)
print("Download strictly complete! Model cached.")
