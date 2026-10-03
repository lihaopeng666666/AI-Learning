from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_all(data_dir):
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, fname))
            docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # 目标最大长度
        chunk_overlap=chunk_overlap, # 重叠部分
        separators=["\n\n", "\n", "。", "；", "，", " ", ""], # 递归切割的优先顺序
        length_function=len # 按字符串的字符数来计算长度
        )
    return splitter.split_documents(docs)

# if __name__ == "__main__":
#     docs = load_all("Data")
#     print(f"原始文档：{len(docs)}")

#     chunks = split_documents(docs)
#     print(f"切分后块数：{len(chunks)}")

#     # 打印前三个块的详情
#     for i, chunk in enumerate(chunks[:3]):
#         print(f"\n--- 块 {i+1} ---")
#         print(f"长度：{len(chunk.page_content)}")
#         print(f"来源：{chunk.metadata.get("source")} 第{chunk.metadata.get("page")}页")
#         print(chunk.page_content[:300])

#     # 统计
#     lengths = [len(c.page_content) for c in chunks]
#     print(f"\n块长度统计")
#     print(f"最短：{min(lengths)}, 最长：{max(lengths)}, 平均：{sum(lengths)/len(lengths):.0f}")

configs = [
    {"chunk_size": 300, "chunk_overlap": 30},
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 800, "chunk_overlap": 80}, 
]
docs = load_all("Data")
for cfg in configs:
    chunks = split_documents(docs, **cfg) # 把字典拆开，变成关键字参数传进去
    lengths = [len(c.page_content) for c in chunks]
    print(f"chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']} "
          f"→ 块数={len(chunks)}, 平均长度={sum(lengths)/len(lengths):.0f}")