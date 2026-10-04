import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
import jieba

load_dotenv()  # 读取env

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
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len
    )
    return splitter.split_documents(docs)

docs = load_all("../Day12/Data")
chunks = split_documents(docs)

# 加载混合检索器（Day14）
bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,  # 需要先加载chunks
    preprocess_func=lambda x: list(jieba.cut(x)),
    k=10
)

embbeddings = HuggingFaceEmbeddings(
    model_name="F:/AI-Learning/Day12/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    # encode_kwrags={"normalize_embeddings": True}
)   # 把文本变成数字向量 越接近回答越准确

vectorstore = Chroma(
    collection_name="civil_codes",
    embedding_function=embbeddings, # 告诉 Chroma 用哪个模型把文本转成向量
    persist_directory="../Day12/chroma_db"
)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5] # 两专家意见各占一半
)

# 加入Reranker
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3) # 重新打分排序，只保留最相关的三篇
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

# RAG问答
client = OpenAI(
    api_key=os.getenv("ALIYUN_BAILIAN_API_KEY"), # 从环境配置中读取密钥
    base_url="https://ws-it60fjx06fvy8l40.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" # 改掉请求地址
)

def rag_qa(question):
    results=retriever.invoke(question) 
    if not results:
        return "未找到相关信息", []
    
    context_parts = []
    citations = []
    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[片段{i+1}]来源：{source} 第{page}页\n{doc.page_content}")
        citations.append(f"{source} 第{page}页")
        
    context = "\n\n".join(context_parts)  # 把所有片段拼成一个完整上下文
    system_prompt = """你是一个土木工程规范助手。请严格根据提供的规范片段回答问题。
要求：
1. 只使用片段中的信息，不要编造。
2. 如果片段中没有答案，直接说“根据提供的规范片段，未找到相关信息”。
3. 回答要简洁、准确，并标注引用的片段编号。
4. 最后列出所有引用来源。""" # 给 AI 设定角色和规则

    user_prompt = f"""
    规范片段：{context}

    问题：{question}

    请回答："""

    # 调用LLM
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1 # 控制输出的随机性，范围通常是0到2
    )

    answer = response.choices[0].message.content
    return answer, citations

#  测试 
if __name__ == "__main__":
    questions = [
        "混凝土强度等级是怎么划分的？",
        "地基基础设计的基本要求是什么？",
        "钢筋保护层厚度有什么规定？",
        "今天天气怎么样？"  # 测试兜底
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"问题：{q}")
        answer, citations = rag_qa(q)
        print(f"\n答案：\n{answer}")
        if citations:
            print(f"\n引用来源：")
            for c in citations:
                print(f"  - {c}")