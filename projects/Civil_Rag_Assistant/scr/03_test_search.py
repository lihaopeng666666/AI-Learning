from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 加载 Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="F:/AI-Learning/Day12/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    # encode_kwrags={"normalize_embeddings": True}
)

# 从磁盘加载已有向量库
vectorstore = Chroma(
    collection_name="civil_codes",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# ======= 测试检索 ======
queries = [
    "GB50010 混凝土强度等级 立方体抗压强度标准值？",
    "地基基础设计的基本要求是什么？",
    "钢筋保护层厚度有什么规定？",
]

for q in queries:
    print(f"\n{"="*60}")
    print(f"查询：{q}")
    results = vectorstore.similarity_search_with_score(q, k=3) # 最相似的前三份资料
    # with_score 不仅要把资料给我，还要把“相似度分数”告诉我（也就是查到的资料和问题的匹配程度）

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Top {i+1} (距离：{score:.4f}) ---")
        print(f"来源：{doc.metadata.get("source")} 第{doc.metadata.get("page")}页")
        print(doc.page_content[:200]) #打印找到的文本内容