import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import os

class Config:
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    DATASET_PATH = "dataset/result/finetuning_data.jsonl"
    OUTPUT_DIR = f"models/{MODEL_NAME.replace('/', '_')}-sft-finetuned-full"
    MAX_LENGTH = 1024

def formatting_prompts_func(example):
    text = (
        f"<|im_start|>system\nBạn là một trợ lý pháp lý chuyên nghiệp.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )
    return text

def main():
    print("--- Starting FULL TRAINING: SFT with Trainer-led device placement ---")
    
    full_dataset = load_dataset("json", data_files=Config.DATASET_PATH, split="train")
    
    train_test_split = full_dataset.train_test_split(test_size=0.05)
    dataset = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    print(f"Dataset loaded and split: {len(dataset['train'])} training samples, {len(dataset['test'])} validation samples.")


    print(f"Loading base model: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    print("Model and tokenizer loaded successfully!")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    
    training_args = SFTConfig(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=3,
        save_strategy="epoch",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=25,
        bf16=True, 
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_length=Config.MAX_LENGTH,
        packing=True,
        report_to='none',
        eval_strategy="epoch", 
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
    )

    print("\n--- STARTING TRAINING ---")
    trainer.train()
    print("--- TRAINING COMPLETE ---\n")
    
    print("Saving final model...")
    final_model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()