from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader # 文档加载器
import os

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件类型: {file_path}")
    return loader.load() #  返回列表

if __name__ == "__main__":
    data_dir = "Data" # 读取当前目录下的Data文件
    all_docs = []
    for fname in os.listdir(data_dir): # 返回Data文件夹下的所有文件名
        path = os.path.join(data_dir, fname) # 把目录和文件名拼成完整路径
        docs = load_document(path)
        all_docs.extend(docs)
        print(f"{fname}: {len(docs)} 页/段")

    print(f"\n总文档数: {len(all_docs)}")
    print(f"\n第一页预览:")
    print(all_docs[0].page_content[:500]) # 取前500个字符
    print("\n元数据:")
    print(all_docs[0].metadata) # 元数据 通常包含 source（文件名）和 page（页码）

    for doc in all_docs:
        print(doc.metadata)


