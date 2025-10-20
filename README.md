# ⚖️ Trợ lý Pháp lý Việt Nam (LVNAI - Legal Virtual-assistant for Vietnamese AI)

Trợ lý Pháp lý Việt Nam là một chatbot AI được xây dựng để trả lời các câu hỏi liên quan đến hệ thống pháp luật của Việt Nam. Dự án sử dụng mô hình RAG (Retrieval-Augmented Generation) kết hợp với các mô hình ngôn ngữ lớn (LLM) để cung cấp câu trả lời chính xác, có trích dẫn nguồn từ cơ sở dữ liệu nội bộ là các văn bản luật.

---

## 🖼️ Demo

<!-- BẠN CÓ THỂ ĐỂ ẢNH HOẶC GIF DEMO ỨNG DỤNG TẠI ĐÂY -->
<!-- Ví dụ: <p align="center"><img src="https://example.com/demo.gif" width="700"></p> -->
<p align="center">
  <img width="1922" height="912" alt="{22A9C1B8-8583-403D-8A7F-803BB199FC2F}" src="https://github.com/user-attachments/assets/f0b8b78b-f334-40bd-ba3b-be20d7a37831" />
</p>
<p align="center">
  <img width="1926" height="915" alt="{7AA1377B-22DF-4000-B541-BBBFBB0F8B50}" src="https://github.com/user-attachments/assets/b336c5aa-f4b4-4a5d-b55b-a6ff073f1987" />
</p>

---

## ✨ Tính năng chính

*   **Tra cứu thông minh**: Người dùng có thể đặt câu hỏi bằng ngôn ngữ tự nhiên để tra cứu thông tin pháp luật.
*   **Trả lời dựa trên nguồn**: Hệ thống sử dụng kỹ thuật RAG, truy xuất các điều luật liên quan từ một Vector Store đã được index trước đó để làm ngữ cảnh cho LLM.
*   **Trích dẫn rõ ràng**: Mọi thông tin trong câu trả lời đều được đính kèm nguồn trích dẫn là mã hiệu của văn bản/điều luật, giúp người dùng dễ dàng tra cứu và xác thực.
*   **Giao diện trực quan**: Xây dựng trên nền tảng Streamlit, cung cấp một giao diện chat thân thiện và dễ sử dụng.
*   **Mở rộng tìm kiếm**: Tích hợp khả năng gợi ý các nguồn tham khảo từ Internet bằng cách sử dụng Gemini, mở rộng phạm vi thông tin cho người dùng.
*   **Quy trình dữ liệu hoàn chỉnh**: Cung cấp đầy đủ các script để thu thập, tiền xử lý, và vector hóa dữ liệu từ các văn bản luật `.doc`/`.docx`.

---

## 🚀 Công nghệ sử dụng

*   **Giao diện người dùng**: Streamlit
*   **Xử lý ngôn ngữ & RAG**: LangChain, Sentence Transformers, FAISS
*   **Mô hình Embedding**: `bkai-foundation-models/vietnamese-bi-encoder`
*   **Mô hình Reranker**: `BAAI/bge-reranker-large`, Flashrank
*   **Mô hình ngôn ngữ (LLM)**: `google/gemini-2.0-flash-exp`, `meta-llama/llama-3-8b-instruct`, `Qwen/Qwen2-7B-Instruct` (thông qua OpenRouter)
*   **Fine-tuning**: Hugging Face Transformers, PEFT, TRL, bitsandbytes
*   **Thu thập dữ liệu**: Selenium, `python-docx`

---

## 📂 Cấu trúc dự án
```
.
├── app.py # File chính để chạy ứng dụng Streamlit
├── dataset/
│ ├── downloads/ # Chứa các file luật (.doc, .docx) tải về
│ ├── downloads_docx/ # Chứa các file luật đã được convert sang .docx
│ ├── result/
│ │ ├── rag_knowledge_base.csv # Dữ liệu luật đã qua xử lý, sẵn sàng cho vector hóa
│ │ └── finetuning_data.jsonl # Dữ liệu hỏi-đáp để fine-tuning
│ └── ...
├── step1/ # Các script cho việc xử lý dữ liệu
│ ├── crawl_data.py # Tải các văn bản luật từ hyperlink trong file docx
│ ├── convert_doc_to_docx.py # Chuyển đổi file .doc sang .docx
│ ├── preprocess_law_data.py # Tách các văn bản luật thành các điều, khoản (chunks)
│ ├── create_vector_store.py # Tạo và lưu trữ vector store từ dữ liệu đã xử lý
│ └── ...
├── step2/ # Các script để xây dựng và thử nghiệm chatbot
│ ├── chatbot_rag_only.py # Phiên bản chatbot RAG chạy trên console
│ └── finetune_llm.py # Script để fine-tuning mô hình LLM (ví dụ: Qwen2)
└── vector_store/
└── faiss_index_v2/ # Thư mục chứa index của FAISS
```

## 🛠️ Hướng dẫn cài đặt và sử dụng

### 1. Yêu cầu

*   Python 3.9 trở lên
*   `pip` để quản lý thư viện
*   API Key từ [OpenRouter](https://openrouter.ai/)

### 2. Cài đặt

#### a. Clone repository:
```bash
git clone https://github.com/nam-htran/LVNAI.git
cd LVNAI
```
#### b. Cài đặt các thư viện cần thiết:
(Bạn cần tạo file requirements.txt từ các thư viện đã import trong dự án, hoặc cài đặt thủ công)
```bash
pip install streamlit langchain-huggingface langchain_community sentence_transformers faiss-cpu litellm python-docx pandas tqdm selenium openpyxl
```
Đối với convert_doc_to_docx.py trên Windows
```
pip install pywin32
```
Đối với fine-tuning
```
pip install torch datasets transformers trl peft bitsandbytes
```
#### c. Cấu hình API Key:
Tạo file `.streamlit/secrets.toml` và thêm API Key của bạn:
```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
```
### 3. Bước 1: Chuẩn bị dữ liệu
Đây là bước quan trọng để xây dựng cơ sở tri thức cho chatbot. Thực hiện các script trong thư mục step1/ theo thứ tự:
#### a. Thu thập dữ liệu (crawl_data.py):
Script này sẽ trích xuất các hyperlink từ file dataset/Danh_muc_Bo_luat_va_Luat_cua_Viet_Nam_2909161046 (1).docx và tự động tải các văn bản luật về thư mục dataset/downloads/.
```
python step1/crawl_data.py
```
Lưu ý: Bạn sẽ được yêu cầu đăng nhập vào trang luatvietnam.vn trên cửa sổ Chrome được mở tự động trước khi quá trình tải bắt đầu.
#### b. Chuẩn hóa định dạng (convert_doc_to_docx.py):
Chuyển đổi tất cả các tệp .doc sang định dạng .docx để dễ dàng xử lý.
```
python step1/convert_doc_to_docx.py
```
#### c. Tiền xử lý văn bản (preprocess_law_data.py):
Đọc nội dung từ các file .docx, làm sạch và chia nhỏ thành các đơn vị kiến thức (từng điều, khoản của luật) và lưu vào file dataset/result/rag_knowledge_base.csv.
```
python step1/preprocess_law_data.py
```
#### d. Tạo Vector Store (create_vector_store.py):
Sử dụng mô hình embedding để vector hóa các đơn vị kiến thức và lưu trữ vào FAISS index tại vector_store/faiss_index_v2/.
```
python step1/create_vector_store.py
```
### 4. Bước 2: Chạy ứng dụng
Sau khi đã có Vector Store, bạn có thể khởi chạy ứng dụng chatbot.
#### a. Chạy ứng dụng Streamlit:
```
streamlit run app.py
```
Mở trình duyệt và truy cập vào địa chỉ http://localhost:8501.
#### b. Chạy phiên bản Console (tùy chọn):
Để thử nghiệm nhanh trên terminal, bạn có thể chạy script chatbot_rag_only.py.
```
python step2/chatbot_rag_only.py
```
