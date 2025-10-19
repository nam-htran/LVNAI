import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

class Config:
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    DATASET_PATH = "dataset/result/finetuning_data.jsonl"
    PROCESSED_DATASET_DIR = "dataset/result/finetuning_packed_data"
    MAX_LENGTH = 1024

def formatting_prompts_func(example):
    text = (
        f"<|im_start|>system\nBạn là một trợ lý pháp lý chuyên nghiệp.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )
    return {"text": text + "<|im_end|>"}

def main():
    print(f"--- Bắt đầu xử lý và đóng gói dataset ---")
    print(f"Model Tokenizer: {Config.MODEL_NAME}")
    print(f"Max Sequence Length: {Config.MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=Config.DATASET_PATH, split="train")
    print(f"Đã tải {len(dataset)} mẫu dữ liệu gốc.")

    dataset = dataset.map(
        formatting_prompts_func,
        num_proc=os.cpu_count()
    )
    print("Đã áp dụng định dạng prompt cho toàn bộ dataset.")

    all_token_ids = []
    for sample in tqdm(dataset, desc="Tokenizing and concatenating"):
        tokenized_output = tokenizer(sample['text'], truncation=False, add_special_tokens=False)
        all_token_ids.extend(tokenized_output['input_ids'])

    print(f"\nTổng cộng có {len(all_token_ids)} tokens trong toàn bộ dataset.")

    packed_examples = []
    total_chunks = len(all_token_ids) // Config.MAX_LENGTH
    for i in tqdm(range(total_chunks), desc="Packing sequences"):
        start_index = i * Config.MAX_LENGTH
        end_index = start_index + Config.MAX_LENGTH
        chunk = all_token_ids[start_index:end_index]
        packed_examples.append({
            'input_ids': chunk,
            'attention_mask': [1] * len(chunk)
        })

    from datasets import Dataset
    processed_dataset = Dataset.from_list(packed_examples)
    print(f"Đã đóng gói thành công thành {len(processed_dataset)} mẫu, mỗi mẫu dài {Config.MAX_LENGTH} tokens.")

    processed_dataset = processed_dataset.shuffle(seed=42)
    processed_dataset.save_to_disk(Config.PROCESSED_DATASET_DIR)

    print(f"\n--- HOÀN TẤT ---")
    print(f"Dataset đã xử lý và lưu tại: '{Config.PROCESSED_DATASET_DIR}'")
    print("Bây giờ bạn có thể chạy script fine-tuning.")

if __name__ == "__main__":
    main()