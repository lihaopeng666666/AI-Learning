import os
import jieba
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

st.set_page_config(page_title="土木规范RAG助手", page_icon="🏗️", layout="wide")

# 缓存资源
@st.cache_resource
def load_retriever():
    """加载检索链，只执行一次"""
    # 加载切分
    def load_all(data_dir):
        docs = []
        for fname in os.listdir(data_dir):
            if fname.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(data_dir, fname))
                docs.extend(loader.load())
        return docs

    def split_documents(docs, chunk_size=500, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len
        )
        return splitter.split_documents(docs)

    docs = load_all(os.path.join(BASE_DIR, "Data"))
    chunks = split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # BM25
    bm25_retriever = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=lambda x: list(jieba.cut(x)),
        k=10
    )

    # 向量
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={"device": "cpu"},
        # encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        collection_name="civil_codes",
        embedding_function=embeddings,
        persist_directory=os.path.join(BASE_DIR, "chroma_db")
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 混合
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )

    # Rerank
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )
    return retriever, chunks

retriever, all_chunks = load_retriever()

# LLM客户端
@st.cache_resource
def load_llm():
    return OpenAI(
        api_key=st.secrets("ALIYUN_BAILIAN_API_KEY"),
        base_url="https://ws-it60fjx06fvy8l40.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )

client = load_llm()

SYSTEM_PROMPT = """你是一个严谨的土木工程规范助手。你的回答必须严格遵守以下规则：

【核心规则】
1. 只使用提供的"规范片段"中的信息回答，禁止使用片段外的知识。
2. 如果片段中没有答案，必须回答："根据提供的规范片段，未找到相关信息。"禁止猜测或编造。
3. 每个结论后面必须标注来源片段编号，格式如 [片段1]。
4. 如果多个片段支持同一结论，标注多个编号，如 [片段1][片段3]。
5. 引用规范原文时，用引号标注。

【禁止行为】
- 禁止编造规范编号、条文号、数值
- 禁止回答与土木工程无关的问题
- 禁止使用"可能""大概""一般来说"等模糊表述"""

# RAG问答
def rag_qa(question):
    docs = retriever.invoke(question)
    if not docs:
        return "未在规范库中找到相关信息，请尝试换一种问法。", [], []

    context_parts = []
    citations = []
    for i, doc in enumerate(docs):
        source = os.path.basename(doc.metadata.get("source", "未知"))
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[片段{i+1}] 来源：{source} 第{page}页\n{doc.page_content}")
        citations.append({
            "index": i + 1,
            "source": source,
            "page": page,
            "preview": doc.page_content[:200]
        })

    context = "\n\n".join(context_parts)
    user_prompt = f"规范片段：\n{context}\n\n问题：{question}\n\n请严格按照规则回答，并标注引用片段编号。"

    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    answer = response.choices[0].message.content
    return answer, citations, docs

# 页面标题
st.title("🏗️ 土木规范 RAG 智能问答助手")
st.caption("基于混合检索 + Rerank + LLM，答案可溯源到规范原文")

# 侧边栏
with st.sidebar: # 把接下来的组件都放在左侧边栏
    st.header("⚙️ 设置")
    top_k = st.slider("检索返回片段数", 1, 10, 5)
    show_sources = st.checkbox("显示检索片段原文", value=True)
    st.divider()
    st.markdown("**技术栈**")
    st.markdown("- 混合检索：BM25 + 向量\n- 精排：BGE Reranker\n- LLM：Qianwen-max \n- 向量库：Chroma")
    st.divider()
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun() # 强制页面重新运行，达到清屏效果

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史信息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): # 根据角色生成不同颜色的聊天气泡
        st.markdown(msg["content"]) # 渲染大模型返回的Markdown格式文本
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 查看引用来源"):
                for c in msg["citations"]:
                    st.markdown(f"**[片段{c['index']}]** {c['source']} 第{c['page']}页")
                    st.caption(c["preview"])

# 输入框
if prompt := st.chat_input("请输入您的规范问题，例如：混凝土强度等级怎么划分？"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索规范并生成答案..."):
            answer, citations, docs = rag_qa(prompt)

        st.markdown(answer)

        if show_sources and citations:
            with st.expander("📚 查看引用来源"):
                for c in citations:
                    st.markdown(f"**[片段{c['index']}]** {c['source']} 第{c['page']}页")
                    st.caption(c["preview"])

        # 保存到历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations
        })