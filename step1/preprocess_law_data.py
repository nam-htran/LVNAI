import os
import re
import pandas as pd
from docx import Document
from tqdm import tqdm

DATA_DIR = "dataset/downloads_docx"
OUTPUT_CSV = "dataset/result/rag_knowledge_base.csv"

def read_docx_text(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Lỗi khi đọc file {os.path.basename(file_path)}: {e}")
        return ""

def clean_raw_text(text):
    text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Trang \d+.*', '', text, flags=re.MULTILINE)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not (len(line.strip()) < 100 and any(keyword in line for keyword in ["QUỐC HỘI", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Văn bản hợp nhất"]) and line.isupper())]
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text, law_number_clean):
    chunks = []
    current_chapter = ""
    current_section = ""
    current_article_number = ""
    current_article_content = ""

    structure_pattern = re.compile(
        r"^(Chương [IVXLCDM\d]+.*?)\n|^(Mục \d+.*?)\n|^(Điều \d+[a-z]?\..*?)$",
        re.MULTILINE | re.IGNORECASE
    )
    
    matches = list(structure_pattern.finditer(text))
    if not matches:
        return []

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content_block = text[start_pos:end_pos].strip()

        chapter_match = match.group(1)
        section_match = match.group(2)
        article_match = match.group(3)

        if chapter_match:
            current_chapter = chapter_match.strip()
            current_section = "" 
        elif section_match:
            current_section = section_match.strip()
        elif article_match:
            if current_article_number and current_article_content:
                process_article(chunks, law_number_clean, current_chapter, current_section, current_article_number, current_article_content)
            
            current_article_number = article_match.strip()
            current_article_content = content_block
    
    if current_article_number and current_article_content:
        process_article(chunks, law_number_clean, current_chapter, current_section, current_article_number, current_article_content)
        
    return chunks

def process_article(chunks, law_number, chapter, section, article_number_raw, article_content):
    article_match_obj = re.match(r"^(Điều \d+[a-z]?)\.?", article_number_raw, re.IGNORECASE)
    if not article_match_obj: return
    article_number_normalized = article_match_obj.group(1).replace(" ", "-")
    
    clause_pattern = re.compile(r"^\s*(\d+\.|[a-z]\))\s", re.MULTILINE)
    clauses = clause_pattern.split(article_content)
    
    if len(clauses) > 2:
        content_remainder = clauses[0]
        i = 1
        while i < len(clauses):
            clause_number = clauses[i]
            clause_content = clauses[i+1]
            full_clause_content = f"{clause_number}{clause_content}".strip()
            
            unique_id = f"{law_number}/{article_number_normalized}/khoan-{clause_number.replace('.', '').replace(')', '')}"
            
            chunks.append({
                "article_id": unique_id,
                "law_number": law_number.replace('-', '/'),
                "chapter": chapter,
                "section": section,
                "article_number": article_number_raw,
                "content": f"{article_number_raw}\n{full_clause_content}"
            })
            i += 2
    else:
        unique_id = f"{law_number}/{article_number_normalized}"
        chunks.append({
            "article_id": unique_id,
            "law_number": law_number.replace('-', '/'),
            "chapter": chapter,
            "section": section,
            "article_number": article_number_raw,
            "content": article_content
        })

def main():
    all_chunks = []
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".docx") and not f.startswith("~")]

    for filename in tqdm(files, desc="Processing Law Documents"):
        file_path = os.path.join(DATA_DIR, filename)
        law_number_clean = os.path.splitext(filename)[0].replace('Luật-', '').replace('Bộ luật-', '')
        
        raw_text = read_docx_text(file_path)
        if not raw_text: continue
        
        cleaned_text = clean_raw_text(raw_text)
        chunks = chunk_text(cleaned_text, law_number_clean)
        all_chunks.extend(chunks)

    df = pd.DataFrame(all_chunks)
    if not df.empty:
        df = df[['article_id', 'law_number', 'chapter', 'section', 'article_number', 'content']]
    
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n--- HOÀN TẤT XỬ LÝ NỘI DUNG ---")
    print(f"Đã xử lý và lưu {len(df)} đoạn luật (chunks) tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()