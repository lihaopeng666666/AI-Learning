import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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

# 加载Embedding模型
print("加载Embedding模型...")
embeddings = HuggingFaceEmbeddings(
    model_name="F:/AI-Learning/Day12/bge-base-zh-v1.5",
    model_kwargs={"device": "cpu"},
    # encode_kwargs={"normalize_embeddings": True} # 把向量归一化，计算相似度更稳定
)
print("模型加载完成")

def vectorstore(data_dir, persist_dir="./chroma_db"):
    docs = load_all(data_dir)
    print(f"原始文档数: {len(docs)}")

    chunks = split_documents(docs)
    print(f"切分后块数: {len(chunks)}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,  # 生成向量
        collection_name="civil_codes",
        persist_directory=persist_dir   # 保存到本地文件夹
    )
    print(f"向量库已保存到: {persist_dir}")
    return vectorstore

if __name__ == "__main__":
    vs = vectorstore("Data")
    print("入库完成")
    