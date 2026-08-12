from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

path = "/share/MY-DAPO/e2e/qwen3-4b-w2g128-asam-pg-fake"

tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

prompt = "What is 1+1?"

inputs = tok(prompt, return_tensors="pt").to("cuda")

out = model.generate(
    **inputs,
    max_new_tokens=128,
)

print(tok.decode(out[0]))