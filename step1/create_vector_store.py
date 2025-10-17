# file: create_vector_store_v2.py
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# --- CONFIG ---
PROCESSED_DATA_PATH = "dataset/result/rag_knowledge_base.csv"
VECTOR_STORE_PATH = "vector_store/faiss_index_v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("Bắt đầu quá trình Indexing...")

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        df = df.dropna(subset=['content'])
        loader = DataFrameLoader(df, page_content_column="content")
        documents = loader.load()
        print(f"Đã load được {len(documents)} documents (Điều luật).")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {PROCESSED_DATA_PATH}. Hãy chạy script preprocess trước.")
        return

    print(f"Đang tải Embedding Model: {EMBEDDING_MODEL} (có thể mất thời gian ở lần đầu)...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Bắt đầu tạo và lưu trữ Vector Store...")
    vector_store = FAISS.from_documents(documents, embedding_model)

    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"\nHoàn tất! Đã lưu Vector Store tại: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main()