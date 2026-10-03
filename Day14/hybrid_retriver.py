import os
import jieba
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever # BM25检索器：关键词搜索
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever

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
    k=5
    )
print("BM25检索器构建完成")

test_queries = [
    "混凝土强度等级怎么划分",
    "GB 50010 保护层厚度",
    "水灰比的要求",
]

for q in test_queries:
    print(f"\n{'='*60}")
    print(f"BM25查询：{q}")
    results = bm25_retriever.invoke(q) # 调用进行检索
    for i, doc in enumerate(results[:3]):
        print(f"\n--- Top {i+1} ---")
        print(f"来源: {doc.metadata.get('source')} 第{doc.metadata.get('page')}页")
        print(doc.page_content[:150])


# # 加载向量检索器
# embbeddings = HuggingFaceEmbeddings(
#     model_name="F:/AI-Learning/Day12/bge-base-zh-v1.5",
#     model_kwargs={"device": "cpu"},
#     # encode_kwrags={"normalize_embeddings": True}
# )   # 把文本变成数字向量 越接近回答越准确

# vectorstore = Chroma(
#     collection_name="civil_codes",
#     embedding_function=embbeddings, # 告诉 Chroma 用哪个模型把文本转成向量
#     persist_directory="../Day12/chroma_db"
# )
# vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 把仓库变成检索工具，每次找最相关的五条

# # 融合检索器
# ensemble_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.5, 0.5] # 两专家意见各占一半
# )

# # 三路对比测试
# test_queries = [
#     "混凝土强度等级怎么划分",
#     "GB 50010 保护层厚度",
#     "地基基础设计的基本要求",
# ]

# for q in test_queries:
#     print(f"\n{"="*70}")
#     print(f"查询:{q}")

#     # 纯向量
#     print("\n---纯向量检索---")
#     vec_results = vector_retriever.invoke(q)
#     for i, doc in enumerate(vec_results[:3]):
#         print(f"  Top{i+1}: {doc.page_content[:80]}...")
        
#     # 纯BM25
#     print("\n--- 纯BM25检索 ---")
#     bm25_results = bm25_retriever.invoke(q)
#     for i, doc in enumerate(bm25_results[:3]):
#         print(f"  Top{i+1}: {doc.page_content[:80]}...")

#     # 混合检索
#     print("\n--- 混合检索 ---")
#     hybrid_results = ensemble_retriever.invoke(q)
#     for i, doc in enumerate(hybrid_results[:5]):
#         print(f"  Top{i+1}: {doc.page_content[:80]}...")

# # 量化对比
# import time

# def evaluate_retriever(retriever, queries, name):
#     start = time.time()
#     all_results = []
#     for q in queries:
#         results = retriever.invoke(q)
#         all_results.append(results)
#     elapsed = time.time() - start
#     print(f"{name}: {len(queries)}次查询耗时{elapsed:.2f}s, 平均{elapsed/len(queries):.2f}s/次")
#     return all_results

# vec_results = evaluate_retriever(vector_retriever, test_queries, "纯向量")
# bm25_results = evaluate_retriever(bm25_retriever, test_queries, "纯BM25")
# hybrid_results = evaluate_retriever(ensemble_retriever, test_queries, "混合检索")

