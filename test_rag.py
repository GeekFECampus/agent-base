from src.utils.rag.chunker import split_text
from src.utils.rag.embedder import get_text_embedding

# 1. 切片测试
text = """RAG检索增强生成，分为文档加载、切片、向量化、向量存储、相似度检索、LLM生成六大步骤。
PGVector是PostgreSQL向量拓展，无需额外部署向量数据库，适合中小型知识库项目。
切片采用固定长度+重叠窗口，避免上下文断裂，提升检索准确度。"""
chunks = split_text(text, chunk_size=200, overlap=50)
print("切片结果：", chunks)

# 2. 向量生成测试
vec = get_text_embedding("什么是PGVector")
print("向量维度：", len(vec)) # 输出384即正常
