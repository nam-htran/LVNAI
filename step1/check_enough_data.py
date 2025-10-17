from docx import Document
import pandas as pd
import os

DOCX_FILE = "dataset/Danh_muc_Bo_luat_va_Luat_cua_Viet_Nam_2909161046 (1).docx"
DOWNLOAD_DIR = "dataset/downloads"

print(f"Đọc file Word: {DOCX_FILE}")
doc = Document(DOCX_FILE)

data = []
for table in doc.tables:
    for row in table.rows:
        if len(row.cells) >= 3:
            stt = row.cells[0].text.strip()
            name = row.cells[1].text.strip()
            so = row.cells[2].text.strip()
            data.append([stt, name, so])

laws = pd.DataFrame(data, columns=['STT', 'Tên Văn bản', 'Số/Văn bản'])
print(f"Đọc được {len(laws)} dòng (bao gồm cả dòng rác)")

exclude_keywords = [
    "tên vb", 
    "số hiệu", 
    "văn bản còn hiệu lực", 
    "văn bản bị sửa đổi", 
    "văn bản hết hiệu lực", 
    "văn bản chưa áp dụng"
]

def is_garbage(row):
    text = (row['Tên Văn bản'] + " " + row['Số/Văn bản']).lower()
    return any(k in text for k in exclude_keywords)

laws_clean = laws[~laws.apply(is_garbage, axis=1)].copy()
laws_clean.reset_index(drop=True, inplace=True)
print(f"Sau khi lọc, còn {len(laws_clean)} dòng hợp lệ.\n")

files = os.listdir(DOWNLOAD_DIR)
files_clean = [os.path.splitext(f)[0].lower() for f in files]

print(f"Thư mục downloads có {len(files_clean)} file.\n")

def check_file_exist_by_so(row):
    so = row['Số/Văn bản'].replace('/', '-').lower()
    return any(so in f for f in files_clean)

laws_clean['File có sẵn'] = laws_clean.apply(check_file_exist_by_so, axis=1)

missing_files = laws_clean[~laws_clean['File có sẵn']]

print("=== TỔNG HỢP KẾT QUẢ ===")
print(laws_clean[['STT', 'Tên Văn bản', 'Số/Văn bản', 'File có sẵn']])
print("\nCác luật chưa có file (nếu có):")
print(missing_files[['STT', 'Tên Văn bản', 'Số/Văn bản']])

os.makedirs("dataset/report", exist_ok=True)
output_path = "dataset/report/check_result.csv"
laws_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nĐã lưu báo cáo tại: {output_path}")
