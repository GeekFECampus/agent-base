from sqlalchemy.ext.asyncio import AsyncSession
from src.db.crud import search_similar_chunk
from src.utils.rag.embedder import get_text_embedding
from src.db.models import DocumentChunk

async def retrieve_knowledge(
    db: AsyncSession,
    kb_id: int,
    query: str,
    top_n: int = 3
) -> list[DocumentChunk]:
    """
    完整检索链路：问题向量化 -> 数据库相似度查询
    """
    emb = get_text_embedding(query)
    chunks = await search_similar_chunk(db, kb_id, emb, top_n)
    return chunks

def format_chunk_context(chunks: list[DocumentChunk]) -> str:
    """把检索到的切片拼接成参考文本，送入Prompt"""
    context = ""
    for idx, chunk in enumerate(chunks, 1):
        context += f"【文档片段{idx}，来源：{chunk.source_name}】\n{chunk.content}\n\n"
    return context.strip()
