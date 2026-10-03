import os
import jieba
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

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
print(f"切分后块数: {len(chunks)}")

bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    preprocess_func=lambda x: list(jieba.cut(x)), # 把一句话切成很多词
    k=10 # 召回数量加大，给Reranker留空间
    )

# 加载向量检索器
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
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # 把仓库变成检索工具，每次找最相关的五条

# 融合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5] # 两专家意见各占一半
)

# 构建reranker
print("加载Reranker模型...")
cross_encoder = HuggingFaceCrossEncoder(model_name="F:/AI-Learning/Day12/bge-base-zh-v1.5")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
print("Reranker加载完成")

# 集成到压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, # 第二阶段的压缩器,这里用重排序器来"压缩"候选集——把不相关的丢掉,只留下最相关的。
    base_retriever=ensemble_retriever # 第一阶段的检索器(
)

test_queries = [
    "混凝土强度等级怎么划分",
    "GB 50010 保护层厚度",
    "地基基础设计的基本要求",
]

for q in test_queries:
    print(f"\n{'='*70}")
    print(f"查询: {q}")

    print("\n--- 混合检索（无Rerank） ---")
    hybrid_results = ensemble_retriever.invoke(q)
    for i, doc in enumerate(hybrid_results[:5]):
        print(f"  Top{i+1}: {doc.page_content[:80]}...")

    print("\n--- 混合检索 + Rerank ---")
    reranked_results = compression_retriever.invoke(q)
    for i, doc in enumerate(reranked_results[:5]):
        print(f"  Top{i+1}: {doc.page_content[:80]}...")

# 评估Reranker效果
import time

def evaluate(retriever, queries, name):
    start = time.time()
    for q in queries:
        retriever.invoke(q)
    elapsed = time.time() - start
    print(f"{name}: 平均 {elapsed/len(queries):.2f}s/次")

evaluate(ensemble_retriever, test_queries, "混合检索")
evaluate(compression_retriever, test_queries, "混合+Rerank")