# 土木规范 RAG 智能问答助手

## 项目简介
基于 RAG 架构的土木工程规范问答系统。用户用自然语言提问，
系统从规范文档中检索相关条文，生成带引用来源的答案。
解决工程师翻规范耗时、LLM 容易编造规范条文的问题。

> 在线 Demo**：[点击访问]()

## 技术架构
用户问题
  → 混合检索（BM25 + BGE向量）
  → RRF融合
  → BGE Reranker精排
  → LLM生成 (ALIYUN)
  → 引用溯源 + 兜底

## 技术栈
| 组件 | 选型 |
|---|---|
| 编排框架 | LangChain |
| 向量库 | Chroma |
| Embedding | BAAI/bge-base-zh-v1.5 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | ALIYUN |
| 关键词检索 | BM25 + jieba |
| 部署 | Streamlit Cloud |

## 核心功能
- 混合检索：BM25 精确匹配 + 向量语义匹配
- Rerank 精排：Cross-Encoder 二次排序
- 引用溯源：答案标注来源文件、页码、片段编号
- 兜底策略：无命中不瞎编，无关问题拒绝回答
- 多轮对话：Streamlit 聊天界面


## 项目结构

```text
civil-rag-assistant/            
├── README.md
├── app.py
├── Data/
│   └── *.pdf
├── chroma_db/
├── bge-base-zh-v1.5
├── src/
│   ├── 01_first_llm_call.py
│   ├── 02_build_vectorstore.py
│   ├── 03_test_search.py
│   ├── 04_rag_qa.py
│   ├── 05_hybrid_retriver.py
│   ├── 06_rag_with_rerank.py
│   ├── 07_rerank_retrieve.py
│   └── 08_rag_final.py

├── images/
│   ├── architecture.png
│   └── demo_screenshot.png
── requirements.txt

1. 快速开始
git clone https://github.com/lihaopeng666666/AI-Learning
cd ai-learning/projects/Civil-Rag-Assistant

2. 构建向量库
python src/02_build_vectorstore.py

3. 命令行问答
python src/08_rag_final.py

4. 启动 Web 应用
streamlit run app.py

5. 安装依赖
pip install -r requirements.txt
