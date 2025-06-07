import os
import sys

current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

import gc
import time
import torch
import wandb
from tqdm import tqdm
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, Dataset
from src.utils.helpers import extract_html_table
from transformers import AutoProcessor, AutoModelForVision2Seq


global processor
        

def format_data(sample, system_message):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message}
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


def load_process_filter_dataset(dataset, max_seq_length, num_train_images, num_test_images, system_message):
    global processor
    ds = load_dataset(dataset, split='train', streaming=True)
    max_html_tokens = max_seq_length - len(processor.tokenizer.tokenize(system_message))
    num_total_needed = num_train_images + num_test_images

    filtered_samples = []
    p_bar = tqdm(total=num_total_needed, desc="Filtering dataset samples")
    for sample in ds:
        processed = process_and_filter_example(sample, max_html_tokens)
        if processed:
            filtered_samples.append(processed)
            p_bar.update(1)
        if len(filtered_samples) >= num_total_needed:
            break
    p_bar.close()

    # Convert to in-memory dataset
    ds_filtered = Dataset.from_list(filtered_samples)

    # Split into train/test
    ds_train = ds_filtered.select(range(num_train_images))
    ds_test = ds_filtered.select(range(num_train_images, num_total_needed))

    return ds_train, ds_test



def process_and_filter_example(example, max_html_tokens):
    global processor
    extracted_table = extract_html_table(example['html_table'])
    token_count = len(processor.tokenizer.tokenize(extracted_table))
    if token_count < max_html_tokens:
        example['html_table'] = extracted_table
        return example
    return None


def main(
        model_name: str, 
        dataset: str, 
        system_message: str,
        max_seq_length: int,
        gradient_accumulation_steps: int, 
        num_train_images: int, 
        num_test_images: int, 
        layers_to_tune: list, 
        experiment_name: str, 
        debug: bool = False,
        resume_id: str = None,
        rewind_step: int = None
    ):

    global processor
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    # Load dataset
    train_dataset, test_dataset = load_process_filter_dataset(
        dataset,
        max_seq_length,
        num_train_images,
        num_test_images,
        system_message
    )

    # Format dataset
    train_dataset = [format_data(x, system_message) for x in train_dataset]
    test_dataset = [format_data(x, system_message) for x in test_dataset]
    
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
        r=16,
        lora_alpha=32,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_seq_length=max_seq_length,
        warmup_steps=10,
        learning_rate=3e-4,
        weight_decay=0.01,
        logging_strategy="steps",
        eval_strategy='steps',
        logging_steps=25,
        save_strategy="steps",
        save_steps=50,
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
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=8
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
        config = {
        "model_name": model_name,
        "dataset": dataset,
        "max_seq_length": max_seq_length,
        "num_train_images": num_train_images
        }

        init_args = {
            "project": "granite-vision",
            "name": experiment_name,
            "config": config
        }
        if resume_id is not None:
            init_args["id"] = resume_id
            if rewind_step is not None:
                init_args["resume_from"] = f"{resume_id}?_step={rewind_step}"
            else:
                init_args["resume"] = "must"
        wandb.init(**init_args)
    else:
        os.environ["WANDB_MODE"] = "offline"

    # Train the model
    # trainer.model.print_trainable_parameters()
    print(f"Memory footprint: {trainer.model.get_memory_footprint() / (1024 ** 3):.2f} GB\n")
    if resume_id is None:
        trainer.evaluate()
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
