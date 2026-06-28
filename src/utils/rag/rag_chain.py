from sqlalchemy.ext.asyncio import AsyncSession
from src.utils.rag.retriever import retrieve_knowledge, format_chunk_context
from src.utils.prompt_manager import prompt_manager

async def build_rag_prompt(
    db: AsyncSession,
    kb_id: int,
    user_query: str,
    history: list = None
) -> str:
    """
    完整RAG链路：检索知识库 + 拼接RAG专用系统提示词
    """
    history = history or []
    # 1. 检索相似文档
    chunks = await retrieve_knowledge(db, kb_id, user_query, top_n=3)
    context_text = format_chunk_context(chunks)
    # 2. 调用PromptManager生成RAG系统prompt
    sys_prompt = prompt_manager.build_rag_chat_prompt(
        user_query=user_query,
        context_docs=context_text,
        history=history
    )
    return sys_prompt
