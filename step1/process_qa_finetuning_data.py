import json
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "thangvip/vietnamese-legal-qa"
OUTPUT_JSONL_PATH = "dataset/result/finetuning_data.jsonl"

def main():
    print(f"--- Bắt đầu xử lý dữ liệu fine-tuning từ '{DATASET_NAME}' ---")

    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        print(f"Tải dữ liệu thành công. Tổng cộng có {len(dataset)} điều luật gốc.")
    except Exception as e:
        print(f"LỖI: Không thể tải dữ liệu từ Hugging Face: {e}")
        return

    finetuning_data = []
    skipped_pairs = 0
    total_pairs = 0

    for row in tqdm(dataset, desc="Đang trích xuất các cặp Hỏi-Đáp"):
        qa_pairs_list = row.get('generated_qa_pairs')

        if not isinstance(qa_pairs_list, list) or not qa_pairs_list:
            continue

        for qa_pair in qa_pairs_list:
            total_pairs += 1
            question = qa_pair.get('question')
            answer = qa_pair.get('answer')

            if not question or not isinstance(question, str) or not question.strip() or \
               not answer or not isinstance(answer, str) or not answer.strip():
                skipped_pairs += 1
                continue
            
            finetuning_data.append({
                "instruction": question,
                "context": "", 
                "output": answer
            })

    print("\n--- KẾT QUẢ XỬ LÝ ---")
    print(f"Tổng số cặp Hỏi-Đáp được tìm thấy: {total_pairs}")
    print(f"Đã xử lý và lưu thành công: {len(finetuning_data)} cặp Hỏi-Đáp")
    if skipped_pairs > 0:
        print(f"Số cặp bị bỏ qua do thiếu dữ liệu: {skipped_pairs}")

    # --- LƯU RA FILE JSONL ---
    with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
        for entry in finetuning_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"\nĐã lưu thành công dữ liệu fine-tuning tại: '{OUTPUT_JSONL_PATH}'")
    print("Bây giờ bạn có thể tiếp tục với bước fine-tuning.")

if __name__ == "__main__":
    main()