import streamlit as st
import os
import re
import urllib.parse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from litellm import completion

# --- CONFIGURATION ---
class AppConfig:
    VECTOR_STORE_PATH = "vector_store/faiss_index_v2"
    EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
    CROSS_ENCODER_MODEL = "BAAI/bge-reranker-large"
    OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"
    RAG_NUM_RETRIEVED_DOCS = 15
    RAG_NUM_RERANKED_DOCS = 5

# --- CACHED RESOURCES ---
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL)
    vector_store = FAISS.load_local(
        AppConfig.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={'k': AppConfig.RAG_NUM_RETRIEVED_DOCS})

@st.cache_resource
def load_reranker():
    return CrossEncoder(AppConfig.CROSS_ENCODER_MODEL)

# --- CORE LOGIC FUNCTIONS ---
def get_rag_context(prompt: str, retriever, reranker) -> str:
    # ... (Hàm này giữ nguyên như cũ, đã hoạt động tốt)
    retrieved_docs = retriever.invoke(prompt)
    if not retrieved_docs: return ""
    rerank_input = [[prompt, doc.page_content] for doc in retrieved_docs]
    scores = reranker.predict(rerank_input)
    doc_scores = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    reranked_docs = [doc for doc, score in doc_scores[:AppConfig.RAG_NUM_RERANKED_DOCS]]
    context_parts = [f"[Nguồn: {doc.metadata.get('article_id', 'Không rõ')}]\n{doc.page_content}" for doc in reranked_docs]
    return "\n\n---\n\n".join(context_parts)

def find_web_sources_with_gemini(question: str) -> str:
    """
    Sử dụng chính Gemini Flash 2.0 để "tìm kiếm" link.
    LƯU Ý: Chức năng này sẽ tạo ra các link không có thật (hallucination).
    """
    st.warning("Đang yêu cầu Gemini tự 'tìm kiếm' link. Kết quả có thể không chính xác.")
    prompt = (
        "You are a web search assistant. Your ONLY task is to find 5 real, existing, and highly relevant Vietnamese web pages for the following query. "
        "Return ONLY a list of Markdown links under the heading `### 📚 Nguồn tham khảo thêm từ Internet:`. "
        f"Query: \"pháp luật Việt Nam về {question}\""
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = completion(
            model=f"openrouter/{AppConfig.OPENROUTER_MODEL}", 
            messages=messages, 
            api_key=st.secrets.get("OPENROUTER_API_KEY"), 
            base_url="https://openrouter.ai/api/v1"
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"> Lỗi khi yêu cầu Gemini tìm link: {e}"

# --- PROMPT & FORMATTING FUNCTIONS ---
# (format_rag_prompt và replace_citations_with_links giữ nguyên như cũ)
def format_rag_prompt(context: str, question: str) -> str:
    system_prompt = (
        "Bạn là một trợ lý pháp lý AI chuyên nghiệp và thân thiện tại Việt Nam..."
    )
    return f"{system_prompt}\n\n**[Bối cảnh]**\n{context}\n\n**[Câu hỏi]**\n{question}"

def replace_citations_with_links(text: str) -> str:
    pattern = r"\[Nguồn: ([\w/-]+)\]"
    def create_search_link(match):
        query = urllib.parse.quote(f'"{match.group(1).split("/")[0].replace("-", "/")}"')
        url = f"https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword={query}"
        return f'<a href="{url}" target="_blank">{match.group(0)}</a>'
    return re.sub(pattern, create_search_link, text)

# --- STREAMLIT UI ---
st.set_page_config(page_title="⚖️ Trợ lý Pháp lý", layout="wide")

with st.spinner("Đang tải tài nguyên..."):
    retriever = load_retriever()
    reranker = load_reranker()

st.title("⚖️ Trợ lý Pháp lý Việt Nam")

if 'history' not in st.session_state:
    st.session_state.history = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}]

for message in st.session_state.history:
    st.chat_message(message["role"]).write(message["content"], unsafe_allow_html=True)

if user_prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.history.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("AI đang phân tích dữ liệu nội bộ..."):
            context = get_rag_context(user_prompt, retriever, reranker)

        if not context.strip():
            rag_answer = "Xin lỗi, tôi không tìm thấy thông tin trong cơ sở dữ liệu nội bộ."
        else:
            with st.spinner("AI đang tổng hợp câu trả lời..."):
                prompt = format_rag_prompt(context, user_prompt)
                messages = [{"role": "user", "content": prompt}]
                try:
                    response = completion(
                        model=f"openrouter/{AppConfig.OPENROUTER_MODEL}", 
                        messages=messages, 
                        api_key=st.secrets.get("OPENROUTER_API_KEY"), 
                        base_url="https://openrouter.ai/api/v1"
                    )
                    rag_answer = response.choices[0].message.content
                except Exception as e:
                    rag_answer = f"Lỗi khi gọi OpenRouter API: {e}"
        
        rag_answer_with_links = replace_citations_with_links(rag_answer)
        message_placeholder.write(rag_answer_with_links, unsafe_allow_html=True)
        
        with st.spinner("Đang yêu cầu AI tìm link trên Internet..."):
            web_sources_list = find_web_sources_with_gemini(user_prompt)

        final_response = f"{rag_answer_with_links}\n\n---\n{web_sources_list}"
        message_placeholder.write(final_response, unsafe_allow_html=True)
        
        st.session_state.history.append({"role": "assistant", "content": final_response})