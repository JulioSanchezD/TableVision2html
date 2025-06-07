import os
import sys
import torch

current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from peft import PeftModel
from huggingface_hub import notebook_login
from transformers import AutoProcessor, AutoModelForVision2Seq


notebook_login()


def main(model_name: str, adapter_path: str) -> None:
    
    # Load processor
    processor = AutoProcessor.from_pretrained(adapter_path)
    print(f"Processor loaded!")

    # Load base model
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map='cpu',
        torch_dtype=torch.float32
    )
    print(f"Base model loaded!")

    # Load adapter onto base model
    model = PeftModel.from_pretrained(
        model, 
        adapter_path, 
        is_trainable=False,
        torch_dtype=torch.float32
    )
    print(f"Adapter loaded from {adapter_path}!")

    # Merge adapter into base model
    model = model.merge_and_unload()
    print(f"Adapter merged into base model!")

    # Upload merged model to Hugging Face
    final_model_name = model_name.split('/')[-1] + '-table2html'
    model.push_to_hub(
        final_model_name, 
        private=False,
        commit_message="Upload LoRA model with merged adapter",
        max_shard_size="1GB"
    )   
    processor.push_to_hub(
        final_model_name, 
        private=False
    )
    print(f"Merged model and processor uploaded to Hugging Face!")
