from time import time
from datasets import load_dataset
from transformers import AutoProcessor
from src.models.granite_vision.transformers_library import LLM as granite_vision


ds = load_dataset('apoidea/pubtabnet-html')
test = ds['validation']

sample = test.take(3)[2]

model_path = "ibm-granite/granite-vision-3.2-2b"
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = granite_vision(
    model_path,
    adapter='lang_only'
)
print('Finished loading merged model')
token_count = 0
start = time()
for token in model.stream(sample['image'], max_new_tokens=1024, query="Convert table to HTML"):
    token_count += 1
    print(token, end='', flush=True)
# print(model.stream(sample['image'], max_new_tokens=1024, query="Convert table to HTML"))
total_time = time() - start
print(f"\nTime taken: {total_time:.2f} seconds")
print(f"Token count: {token_count}")
print(f"Tokens per second: {token_count / total_time:.2f}")
