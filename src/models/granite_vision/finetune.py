import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

import gc
import time
import torch
import wandb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoProcessor, AutoModelForVision2Seq


SYSTEM_MESSAGE = "Convert table to HTML"
global processor
        

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_MESSAGE}
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample['image']
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": sample['html_table']
                }
            ],
        },
    ]


def collate_fn(examples):
    global processor
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    image_inputs = []
    for example in examples:
        image = example[1]["content"][0]["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        # print(f"HTML size: {len(processor.tokenizer.tokenize(example[2]['content'][0]['text']))} tokens")  # debugs max seq length
        image_inputs.append([image])
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    assistant_tokens = processor.tokenizer("<|assistant|>", return_tensors="pt")["input_ids"][0]
    eos_token = processor.tokenizer("<|end_of_text|>", return_tensors="pt")["input_ids"][0]

    for i in range(batch["input_ids"].shape[0]):
        apply_loss = False
        for j in range(batch["input_ids"].shape[1]):
            if not apply_loss:
                labels[i][j] = -100
            if (j >= len(assistant_tokens) + 1) and torch.all(
                batch["input_ids"][i][j + 1 - len(assistant_tokens) : j + 1] == assistant_tokens
            ):
                apply_loss = True
            if batch["input_ids"][i][j] == eos_token:
                apply_loss = False

    batch["labels"] = labels
    return batch


def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def filter_fn(examples, processor, max_seq_length):
    token_counts = [len(processor.tokenizer.tokenize(html)) for html in examples['html_table']]
    return [cnt < max_seq_length for cnt in token_counts]


def main(
        model_name: str, 
        dataset: str, 
        max_seq_length: int, 
        num_train_images: int, 
        num_test_images: int, 
        layers_to_tune: list, 
        experiment_name: str, 
        debug: bool = False
    ):

    global processor
    # Load dataset
    ds = load_dataset(dataset)['train']

    # Filter dataset for memory requirements
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    ds = ds.filter(
        filter_fn, 
        fn_kwargs={'processor': processor, 'max_seq_length': max_seq_length - len(processor.tokenizer.tokenize(SYSTEM_MESSAGE))}, 
        batched=True, 
        batch_size=1000, 
        num_proc=16
    )

    # Split dataset into train and test
    train_test = ds.select(range(num_train_images)).train_test_split(test_size=0.2, seed=42)

    # Format dataset
    train_dataset = [format_data(x) for x in train_test['train']]
    test_dataset = [format_data(x) for x in train_test['test'].select(range(num_test_images))]
    
    # Load Model and tokenizer
    clear_memory()
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # Setup LoRA
    target_modules = []
    for layer_type in layers_to_tune:
        target_modules.extend(
            name for name, _ in model.named_modules()
            if (layer_type in name) 
            and '_proj' in name
        )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        use_dora=True,
        init_lora_weights="gaussian"
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir=f"src/models/{model_name.split('/')[-1].replace('-', '_', 1).split('-')[0]}/checkpoints/{experiment_name}",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_seq_length=max_seq_length,
        warmup_steps=10,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_strategy="steps",
        eval_strategy='steps',
        logging_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        greater_is_better=False,
        load_best_model_at_end=True,
        optim="adamw_torch_fused",
        bf16=True,
        push_to_hub=False,
        report_to="wandb" if not debug else "none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True}
    )

    # Setup Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer
    )

    # Login to Weights & Biases
    if not debug:
        wandb.login()
        run = wandb.init(
            project=f"granite-vision",
            name=experiment_name,
            config={
                "model_name": model_name,
                "dataset": dataset,
                "max_seq_length": max_seq_length,
                "num_train_images": num_train_images
            }
        )
    else:
        os.environ["WANDB_MODE"] = "offline"

    # Train the model
    trainer.model.print_trainable_parameters()
    print(f"Memory footprint: {trainer.model.get_memory_footprint() / (1024 ** 3):.2f} GB\n")
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()