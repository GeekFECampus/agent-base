from typing import List, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.models import User, ChatSession, ChatMessage
from src.api.schema import ChatHistoryItem
from src.core.logger import log
from pgvector.sqlalchemy import Vector
from src.db.models import KnowledgeBase, DocumentChunk

# ===================== User 用户操作 =====================
async def get_user_by_token(db: AsyncSession, token: str) -> Optional[User]:
    stmt = select(User).where(User.token == token)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, token: str) -> User:
    user = User(token=token)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    log.info(f"新建用户，token:{token}")
    return user

# ===================== 会话操作 =====================
async def create_chat_session(db: AsyncSession, user_id: int, session_name: str = "默认会话") -> ChatSession:
    session = ChatSession(user_id=user_id, session_name=session_name)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session

async def get_session_by_id(db: AsyncSession, session_id: int, user_id: int) -> Optional[ChatSession]:
    stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
    res = await db.execute(stmt)
    return res.scalar_one_or_none()

# ===================== 对话消息操作 =====================
async def add_chat_message(db: AsyncSession, session_id: int, role: str, content: str) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg

async def get_session_history(db: AsyncSession, session_id: int) -> List[ChatHistoryItem]:
    """读取会话全部历史，转为接口标准ChatHistoryItem格式"""
    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    result = await db.execute(stmt)
    msg_list = result.scalars().all()
    history = [ChatHistoryItem(role=m.role, content=m.content) for m in msg_list]
    return history

# ========== 知识库操作 ==========
async def create_kb(db: AsyncSession, name: str, desc: str = "") -> KnowledgeBase:
    kb = KnowledgeBase(name=name, desc=desc)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb

async def list_all_kb(db: AsyncSession) -> List[KnowledgeBase]:
    res = await db.execute(select(KnowledgeBase).order_by(KnowledgeBase.id))
    return res.scalars().all()

async def get_kb_by_id(db: AsyncSession, kb_id: int) -> Optional[KnowledgeBase]:
    res = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    return res.scalar_one_or_none()

# ========== 文档切片向量入库 ==========
async def insert_document_chunk(
    db: AsyncSession,
    kb_id: int,
    content: str,
    embedding: Vector,
    source_name: str
) -> DocumentChunk:
    chunk = DocumentChunk(
        kb_id=kb_id,
        content=content,
        embedding=embedding,
        source_name=source_name
    )
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk

# 批量插入切片
async def batch_insert_chunks(db: AsyncSession, chunk_list: List[dict]):
    objs = [
        DocumentChunk(
            kb_id=item["kb_id"],
            content=item["content"],
            embedding=item["embedding"],
            source_name=item["source_name"]
        )
        for item in chunk_list
    ]
    db.add_all(objs)
    await db.commit()

# ========== 向量相似度检索 ==========
async def search_similar_chunk(
    db: AsyncSession,
    kb_id: int,
    query_embedding: List[float],
    top_n: int = 3
) -> List[DocumentChunk]:
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.kb_id == kb_id)
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
        .limit(top_n)
    )
    res = await db.execute(stmt)
    return res.scalars().all()
