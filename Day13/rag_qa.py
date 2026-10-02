import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # 读取env

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

client = OpenAI(
    api_key=os.getenv("ALIYUN_BAILIAN_API_KEY"),
    base_url="https://ws-it60fjx06fvy8l40.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

def rag_qa(question, k=3, score_threshold=1.2): # score_threshold 距离阈值 过滤不相关结果
    results=vectorstore.similarity_search_with_score(question, k=k) # 把用户问题转向量，然后在 Chroma 里找最相似的 k 个文本片段
    filtered = [(doc, score) for (doc, score) in results if score < score_threshold]
    if not filtered:
        return "未在规范库中找到相关信息，请尝试换一种问法或上传相关文档。", []
    context_parts = []
    citations = []
    for i, (doc, score) in enumerate(filtered):
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