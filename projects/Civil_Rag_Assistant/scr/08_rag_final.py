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
print(f"切分后块数：{len(chunks)}")

# 给每个chunk加唯一ID，方便引用溯源
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

# 加载混合检索器（Day14/15）
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
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5) # 重新打分排序，只保留最相关的三篇
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)

# 引用溯源：构造带编号的上下文
def build_context(docs):
    context_parts = []
    citations = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        source_name = os.path.basename(source)
        context_parts.append(f"[片段{i+1}]来源：{source_name} 第{page}页\n{doc.page_content}")
        citations.append({
            "index": i + 1,
            "source": source_name,
            "page": page,
            "chunk_id": chunk_id,
            "preview": doc.page_content[:100]
        })
    return "\n\n".join(context_parts), citations

# 兜底策略
def check_retrieval_quality(docs, min_docs=1):
    """检查检索结果是否足够回答"""
    if not docs or len(docs) < min_docs:
        return False, "未在规范库中找到相关内容"

    # 检查是否太短（可能是噪声）
    total_len = sum(len(d.page_content) for d in docs)
    if total_len < 50:
        return False, "检索到的内容过短, 无法回答"

    return True, ""

def is_question_related(question, docs):
    """简单判断问题是否和检索内容相关"""
    # 用关键词重叠做粗饰
    q_words = set(jieba.cut(question))
    for doc in docs:
        d_words = set(jieba.cut(doc.page_content))
        over_lap = q_words & d_words # 找重叠词
        meaningful = [w for w in over_lap if len(w) > 1] # 过滤有效词 去掉"的", "是" 这类的
        # 判断是否相关
        if len(meaningful) >= 2:
            return True
    return False


# RAG问答
client = OpenAI(
    api_key=os.getenv("ALIYUN_BAILIAN_API_KEY"), # 从环境配置中读取密钥
    base_url="https://ws-it60fjx06fvy8l40.cn-beijing.maas.aliyuncs.com/compatible-mode/v1" # 改掉请求地址
)

# 优化prompt
SYSTEM_PROMPT = """你是一个严谨的土木工程规范助手。你的回答必须严格遵守以下规则：

【核心规则】
1. 只使用提供的"规范片段"中的信息回答，禁止使用片段外的知识。
2. 如果片段中没有答案，必须回答："根据提供的规范片段，未找到相关信息。"禁止猜测或编造。
3. 每个结论后面必须标注来源片段编号，格式如 [片段1]。
4. 如果多个片段支持同一结论，标注多个编号，如 [片段1][片段3]。
5. 引用规范原文时，用引号标注。

【回答格式】
- 先给出简洁答案
- 再列出依据的片段编号
- 如果涉及数值、等级、限值，必须准确引用，不得四舍五入或改写

【禁止行为】
- 禁止编造规范编号、条文号、数值
- 禁止回答与土木工程无关的问题
- 禁止使用"可能""大概""一般来说"等模糊表述"""


def rag_qa(question):
    # 检索
    docs = retriever.invoke(question)

    # 兜底检查
    ok, msg = check_retrieval_quality(docs)
    if not ok:
        return {"answer": msg, "citations": [], "status": "no_result"}

    if not is_question_related(question, docs):
        return {"answer": "您的问题似乎与规范库内容无关，请提问土木工程规范相关的问题。",
                "citations": [],
                "status": "irrelevant"
        }

    # 构造上下文
    context, citations = build_context(docs)

    # 调用LLM
    user_prompt = f"""
    规范片段：{context}

    问题：{question}

    请严格按照规则回答，并标注引用片段编号。"""

    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1 # 控制输出的随机性，范围通常是0到2
    )
    answer = response.choices[0].message.content
    
    #检查LLM是否真的标注了引用
    has_citation = any(f"[片段{c["index"]}]" in answer for c in citations)
    if not has_citation and "未找到" not in answer:
        answer += "\n\n(注: 本次回答为明确标注片段编号，请核实原文。)"
    return {
        "answer": answer,
        "citations": citations,
        "status": "success"
    }
    

#  测试 
if __name__ == "__main__":
    test_questions = [
        ("混凝土强度等级是怎么划分的？", "success"),
        ("地基基础设计的基本要求是什么？", "success"),
        ("钢筋保护层厚度有什么规定？", "success"),
        ("今天天气怎么样？", "irrelevant"),
        ("如何做红烧肉？", "irrelevant"),
    ]
    
    for q, expected in test_questions:
        print(f"\n{'='*70}")
        print(f"问题：{q}")
        result = rag_qa(q)
        print(f"状态：{result['status']}")
        print(f"答案：\n{result['answer']}")
        if result["citations"]:
            print(f"\n引用来源：")
            for c in result["citations"]:
                print(f"  [片段{c['index']}] {c['source']} 第{c['page']}页")