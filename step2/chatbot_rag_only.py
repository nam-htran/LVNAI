import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

class Config:
    DEBUG = True
    VECTOR_STORE_PATH = "vector_store/faiss_index_v2" 
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
    LLM_MODEL = "meta-llama/llama-3-8b-instruct"
    CANDIDATES_TO_RETRIEVE = 20
    TOP_N_AFTER_RERANK = 5

if not Path(Config.VECTOR_STORE_PATH).exists():
    print(f"LỖI: Không tìm thấy thư mục Vector Store tại '{Config.VECTOR_STORE_PATH}'.")
    print("Vui lòng chạy các script `preprocess_law_data.py` và `create_vector_store.py` trước.")
    exit()

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("LỖI: Biến môi trường OPENROUTER_API_KEY không được tìm thấy.")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

print("Đang tải Embedding Model và Vector Store...")
embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
vector_store = FAISS.load_local(Config.VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
base_retriever = vector_store.as_retriever(search_kwargs={'k': Config.CANDIDATES_TO_RETRIEVE})
print("Tải thành công!")

print(f"Đang tải mô hình Reranker: {Config.RERANKER_MODEL}...")
compressor = FlashrankRerank(top_n=Config.TOP_N_AFTER_RERANK, model=Config.RERANKER_MODEL)
print("Tải Reranker thành công!")

final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

prompt_template = """Bạn là một trợ lý AI pháp lý chuyên nghiệp của Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng DỰA HOÀN TOÀN vào các thông tin, ngữ cảnh được cung cấp dưới đây.

**Ngữ cảnh (Các trích đoạn luật liên quan):**
---
{context}
---

**Câu hỏi của người dùng:**
{question}

**Hướng dẫn trả lời:**
1.  Đọc kỹ ngữ cảnh và chỉ sử dụng thông tin từ đó để tổng hợp câu trả lời.
2.  Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nói rõ: "Dựa trên các văn bản luật được cung cấp, tôi không tìm thấy thông tin chính xác để trả lời câu hỏi này."
3.  Trích dẫn nguồn một cách rõ ràng sau mỗi luận điểm bằng cách sử dụng metadata của ngữ cảnh, ví dụ: `[Nguồn: article_id]`.
4.  Câu trả lời phải rõ ràng, mạch lạc, chuyên nghiệp và bằng tiếng Việt.

**Câu trả lời của bạn:**
"""

def get_rag_response(query: str) -> str:
    print("Đang truy xuất và xếp hạng lại văn bản...")
    reranked_docs = final_retriever.invoke(query)

    if Config.DEBUG:
        print("\n--- KẾT QUẢ TÌM KIẾM (SAU KHI RERANK) ---")
        if not reranked_docs:
            print("Không tìm thấy tài liệu nào sau khi rerank.")
        else:
            for i, doc in enumerate(reranked_docs):
                score = doc.metadata.get('relevance_score', 'N/A')
                print(f"[{i+1}] Nguồn: {doc.metadata.get('article_id', 'N/A')} (Điểm: {score})")
                content_snippet = doc.page_content[:150].replace('\n', ' ')
                print(f"    Nội dung: {content_snippet}...")
        print("-------------------------------------------\n")

    context_parts = [f"[Nguồn: {doc.metadata.get('article_id', 'Không rõ')}]\n{doc.page_content}" for doc in reranked_docs]
    context = "\n\n".join(context_parts)
    
    prompt = prompt_template.format(context=context, question=query)
    
    print("Đang gửi prompt đến LLM để tạo câu trả lời...")
    try:
        response = client.chat.completions.create(
            model=Config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"\n[Lỗi]: Đã có lỗi xảy ra khi gọi LLM: {e}"

def main():
    print("\n--- Chatbot Pháp luật Việt Nam (Correct LCEL Version) ---")
    print(f"Sử dụng LLM: {Config.LLM_MODEL}")
    print(f"Chế độ DEBUG: {'Bật' if Config.DEBUG else 'Tắt'}")
    print("Gõ 'exit' để thoát.")
    
    while True:
        query = input("\n[Bạn hỏi]: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        print("\nĐang xử lý yêu cầu...")
        answer = get_rag_response(query)
        print("\n[AI trả lời]:")
        print(answer)

if __name__ == "__main__":
    main()