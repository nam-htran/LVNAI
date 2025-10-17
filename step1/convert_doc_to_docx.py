import os
import win32com.client as win32
from tqdm import tqdm

SOURCE_DIR = os.path.abspath("dataset/downloads")
TARGET_DIR = os.path.abspath("dataset/downloads_docx")

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Đã tạo thư mục mới: {TARGET_DIR}")

    word = None
    try:
        word = win32.Dispatch("Word.Application")
        word.Visible = False

        all_files = os.listdir(SOURCE_DIR)
        
        doc_files = [f for f in all_files if f.lower().endswith(".doc") and not f.startswith('~')]
        
        print(f"\nBắt đầu chuyển đổi {len(doc_files)} file .doc sang .docx...")
        for filename in tqdm(doc_files, desc="Converting .doc files"):
            source_path = os.path.join(SOURCE_DIR, filename)
            target_filename = os.path.splitext(filename)[0] + ".docx"
            target_path = os.path.join(TARGET_DIR, target_filename)

            if os.path.exists(target_path):
                continue

            try:
                doc = word.Documents.Open(source_path)
                doc.SaveAs(target_path, FileFormat=16)
                doc.Close()
            except Exception as e:
                print(f"\nLỗi khi chuyển đổi file {filename}: {e}")
        
        docx_files = [f for f in all_files if f.lower().endswith(".docx") and not f.startswith('~')]
        print(f"\nBắt đầu sao chép {len(docx_files)} file .docx có sẵn...")
        from shutil import copy2
        for filename in tqdm(docx_files, desc="Copying .docx files"):
            source_path = os.path.join(SOURCE_DIR, filename)
            target_path = os.path.join(TARGET_DIR, filename)
            if not os.path.exists(target_path):
                 copy2(source_path, target_path)

    finally:
        if word:
            word.Quit()

    print("\n--- HOÀN TẤT CHUẨN HÓA ĐỊNH DẠNG ---")
    print(f"Toàn bộ văn bản đã được lưu dưới dạng .docx tại thư mục: {TARGET_DIR}")

if __name__ == "__main__":
    main()