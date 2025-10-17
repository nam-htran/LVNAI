# file: convert_doc_to_docx.py
import os
import win32com.client as win32
from tqdm import tqdm

# --- CONFIG ---
# Thư mục chứa các file .doc và .docx gốc
SOURCE_DIR = os.path.abspath("dataset/downloads")
# Thư mục mới để chứa tất cả các file sau khi đã được chuẩn hóa sang .docx
TARGET_DIR = os.path.abspath("dataset/downloads_docx")

def main():
    """
    Tự động chuyển đổi tất cả file .doc sang .docx bằng Microsoft Word.
    Các file .docx sẵn có sẽ được sao chép qua.
    """
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Đã tạo thư mục mới: {TARGET_DIR}")

    word = None
    try:
        # Khởi động ứng dụng Word chạy ngầm
        word = win32.Dispatch("Word.Application")
        word.Visible = False

        all_files = os.listdir(SOURCE_DIR)
        
        # Lọc ra các file văn bản cần xử lý
        doc_files = [f for f in all_files if f.lower().endswith(".doc") and not f.startswith('~')]
        
        print(f"\nBắt đầu chuyển đổi {len(doc_files)} file .doc sang .docx...")
        for filename in tqdm(doc_files, desc="Converting .doc files"):
            source_path = os.path.join(SOURCE_DIR, filename)
            # Tạo tên file mới với đuôi .docx
            target_filename = os.path.splitext(filename)[0] + ".docx"
            target_path = os.path.join(TARGET_DIR, target_filename)

            if os.path.exists(target_path):
                continue # Bỏ qua nếu file đã được chuyển đổi

            try:
                doc = word.Documents.Open(source_path)
                # FileFormat=16 tương ứng với định dạng .docx
                doc.SaveAs(target_path, FileFormat=16)
                doc.Close()
            except Exception as e:
                print(f"\nLỗi khi chuyển đổi file {filename}: {e}")
        
        # Sao chép các file .docx đã có sẵn
        docx_files = [f for f in all_files if f.lower().endswith(".docx") and not f.startswith('~')]
        print(f"\nBắt đầu sao chép {len(docx_files)} file .docx có sẵn...")
        from shutil import copy2
        for filename in tqdm(docx_files, desc="Copying .docx files"):
            source_path = os.path.join(SOURCE_DIR, filename)
            target_path = os.path.join(TARGET_DIR, filename)
            if not os.path.exists(target_path):
                 copy2(source_path, target_path)

    finally:
        # Đảm bảo ứng dụng Word luôn được đóng, kể cả khi có lỗi
        if word:
            word.Quit()

    print("\n--- HOÀN TẤT CHUẨN HÓA ĐỊNH DẠNG ---")
    print(f"Toàn bộ văn bản đã được lưu dưới dạng .docx tại thư mục: {TARGET_DIR}")

if __name__ == "__main__":
    main()