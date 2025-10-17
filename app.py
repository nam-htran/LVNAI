import streamlit as st
import torch
import time
import os
import re
import urllib.parse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from litellm import completion


class AppConfig:
    FINETUNED_MODEL_PATH = "models/Qwen_Qwen2-0.5B-Instruct-sft-finetuned-full/final_model"
    VECTOR_STORE_PATH = "vector_store/faiss_index_v2"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # MÔ HÌNH CHO CHẾ ĐỘ RAG
    OPENROUTER_MODEL = "tngtech/deepseek-r1t2-chimera:free"
    # MÔ HÌNH RIÊNG BIỆT ĐỂ TÌM KIẾM WEB
    WEB_SEARCH_MODEL = "google/gemini-2.0-flash-exp:free"

@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        AppConfig.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={'k': 10})

@st.cache_resource
def load_reranker():
    return CrossEncoder(AppConfig.CROSS_ENCODER_MODEL)

@st.cache_resource
def load_finetuned_pipeline():
    if not os.path.isdir(AppConfig.FINETUNED_MODEL_PATH):
        return None
    model = AutoModelForCausalLM.from_pretrained(
        AppConfig.FINETUNED_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(AppConfig.FINETUNED_MODEL_PATH)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def auto_format_response(text: str) -> str:
    text = re.sub(r'(?<!\n)\n?(###\s)', r'\n\n\1', text)
    text = re.sub(r'(?<!\n)\n?(\d+\.\s)', r'\n\1', text)
    text = re.sub(r'(?<!\n)\n?(-\s)', r'\n\1', text)
    return text.strip()

def format_rag_prompt(context, question, model_type='qwen'):
    system_prompt_content = (
        "Bạn là một trợ lý pháp lý AI chuyên nghiệp và thân thiện tại Việt Nam."
        "Nhiệm vụ của bạn là cung cấp câu trả lời pháp lý chính xác, dễ hiểu dựa trên ngữ cảnh được cung cấp."
        "**QUY TẮC ĐỊNH DẠNG:**"
        "1. Sử dụng các tiêu đề Markdown (`###`) có biểu tượng emoji để phân chia các phần rõ ràng."
        "2. **In đậm** tất cả các thuật ngữ pháp lý quan trọng và các kết luận chính."
        "3. Sử dụng dấu đầu dòng (`-` hoặc `*`) để liệt kê các điểm."
        "4. Dùng blockquote (`>`) để nhấn mạnh các lưu ý đặc biệt quan trọng."
        "**QUY TẮC TRÍCH DẪN NGUỒN (RẤT QUAN TRỌNG):**"
        "1. Sau mỗi luận điểm hoặc thông tin pháp lý bạn đưa ra, bạn **BẮT BUỘC** phải trích dẫn nguồn gốc của thông tin đó."
        "2. Nguồn được cung cấp trong phần Bối cảnh, có dạng `[Nguồn: article_id]`."
        "Hãy trả lời câu hỏi của người dùng dựa trên các điều luật được cung cấp dưới đây."
        "Nếu thông tin không có trong luật, hãy nói rõ rằng bạn không tìm thấy thông tin."
    )
    if model_type == 'qwen':
        return (
            f"<|im_start|>system\n{system_prompt_content}<|im_end|>\n"
            f"<|im_start|>user\n[Bối cảnh từ các điều luật]\n{context}\n\n[Câu hỏi]\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return f"{system_prompt_content}\n\n[Bối cảnh từ các điều luật]\n{context}\n\n[Câu hỏi]\n{question}"

def get_rag_context(prompt, retriever, reranker):
    retrieved_docs = retriever.invoke(prompt)
    rerank_input = [[prompt, doc.page_content] for doc in retrieved_docs]
    scores = reranker.predict(rerank_input)
    doc_scores = list(zip(retrieved_docs, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in doc_scores[:3] if score > 0.1]
    context_parts = []
    for doc in reranked_docs:
        source_info = doc.metadata.get('article_id', 'Không rõ nguồn')
        context_parts.append(f"[Nguồn: {source_info}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)

def replace_citations_with_links(text: str) -> str:
    pattern = r"\[Nguồn: ([\w/-]+)\]"
    def create_search_link(match):
        article_id = match.group(1)
        law_number_searchable = article_id.split('/')[0].replace('-', '/')
        query = urllib.parse.quote(law_number_searchable)
        url = f"https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword={query}"
        return f'<a href="{url}" target="_blank">{match.group(0)}</a>'
    return re.sub(pattern, create_search_link, text)

# --- HÀM MỚI: Chỉ tìm kiếm và trả về danh sách link ---
def find_web_sources(question: str) -> str:
    """Sử dụng AI để tìm kiếm và chỉ trả về danh sách các link nguồn."""
    prompt = (
        "You are a legal research assistant. Your ONLY task is to find the top 5 most reputable Vietnamese web pages for the following query. "
        "ONLY return a list of Markdown links under the heading `### 📚 Nguồn tham khảo thêm từ Internet:`. "
        "Do not add any other explanation or summary. Prioritize sources from `thuvienphapluat.vn`, `luatvietnam.vn`, `vbpl.vn`, and major news sites."
        f"\nQuery: \"{question}\""
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = completion(
            model=f"openrouter/{AppConfig.WEB_SEARCH_MODEL}",
            messages=messages,
            api_key=st.secrets["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"\n> Lỗi khi tìm kiếm nguồn online: {e}"

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="⚖️ Trợ lý Pháp lý", layout="wide")

with st.spinner("Đang tải các mô hình nền, vui lòng chờ..."):
    finetuned_pipeline = load_finetuned_pipeline()
    retriever = load_retriever()
    reranker = load_reranker()

st.title("⚖️ Trợ lý Pháp lý Việt Nam")
st.sidebar.title("Chế độ")

mode_options = ["RAG Only (OpenRouter)", "RAG + LLM Fine-tuned"]
if finetuned_pipeline is None:
    st.sidebar.warning(f"Không tìm thấy mô hình fine-tuned tại:\n{AppConfig.FINETUNED_MODEL_PATH}\nChế độ 'RAG + LLM Fine-tuned' đã bị vô hiệu hóa.")
    mode_options = ["RAG Only (OpenRouter)"]
mode = st.sidebar.radio("Chọn chế độ bạn muốn sử dụng:", mode_options)

if 'history' not in st.session_state:
    st.session_state.history = {}
if mode not in st.session_state.history:
    st.session_state.history[mode] = []
    st.session_state.history[mode].append({"role": "assistant", "content": "Tôi có thể giúp gì cho bạn về luật Việt Nam?"})

for message in st.session_state.history[mode]:
    st.chat_message(message["role"]).write(message["content"], unsafe_allow_html=True)

user_prompt = st.chat_input("Nhập câu hỏi của bạn...")

if user_prompt and user_prompt.strip():
    st.session_state.history[mode].append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        rag_answer = ""
        
        with st.spinner("AI đang phân tích dữ liệu nội bộ..."):
            context = get_rag_context(user_prompt, retriever, reranker)
            if not context or not context.strip():
                rag_answer = "Xin lỗi, tôi không tìm thấy thông tin trong cơ sở dữ liệu nội bộ để trả lời câu hỏi này."
            else:
                if mode == "RAG Only (OpenRouter)":
                    prompt_for_api = format_rag_prompt(context, user_prompt, model_type='deepseek')
                    messages = [{"role": "user", "content": prompt_for_api}]
                    try:
                        response = completion(model=f"openrouter/{AppConfig.OPENROUTER_MODEL}", messages=messages, api_key=st.secrets["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
                        rag_answer = response.choices[0].message.content.replace("undefined", "")
                    except Exception as e:
                        rag_answer = f"Lỗi khi gọi OpenRouter API: {e}"
                else: # Chế độ RAG + LLM Fine-tuned
                    formatted_prompt = format_rag_prompt(context, user_prompt, model_type='qwen')
                    response_data = finetuned_pipeline(formatted_prompt, max_new_tokens=1024, pad_token_id=finetuned_pipeline.tokenizer.eos_token_id)
                    rag_answer = response_data[0]['generated_text'].split("<|im_start|>assistant\n")[1].strip().replace("undefined", "")
        
        # Hiển thị câu trả lời RAG (đã chuyển thành link) trước
        rag_answer_with_links = replace_citations_with_links(rag_answer)
        message_placeholder.write(rag_answer_with_links, unsafe_allow_html=True)
        
        # Sau đó, tìm kiếm thêm các nguồn online
        with st.spinner("Đang tìm các nguồn tham khảo thêm từ Internet..."):
            web_sources_list = find_web_sources(user_prompt)

        # Kết hợp câu trả lời cuối cùng
        final_response = f"{rag_answer_with_links}\n\n---\n{web_sources_list}"
        message_placeholder.write(final_response, unsafe_allow_html=True)
        
        st.session_state.history[mode].append({"role": "assistant", "content": final_response})