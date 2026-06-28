from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
# from src.core.db import Base

Base = declarative_base()

# --------------------------知识点分割线---------------------------
# 1. 外键关联：ChatMessage -> ChatSession 一对多，一个会话多条消息
# 2. relationship：ORM关联查询，直接通过对象获取关联数据，无需手写JOIN SQL
# 3. default=datetime.utcnow：自动记录创建时间，时间统一使用UTC避免时区混乱
# ----------------------------------------------------------------

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    token = Column(String(255), unique=True, nullable=False, index=True)
    create_time = Column(DateTime, default=datetime.utcnow)

    # 关联会话
    sessions = relationship("ChatSession", back_populates="user")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_name = Column(String(128), default="默认会话")
    create_time = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(16), nullable=False) # user / assistant
    content = Column(Text, nullable=False)
    create_time = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

# 原有 User / ChatSession / ChatMessage 省略

# 知识库文档库
class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, comment="知识库名称")
    desc = Column(Text, comment="知识库描述")
    created_at = Column(DateTime, server_default=func.now())

# 文档切片向量表
class DocumentChunk(Base):
    __tablename__ = "document_chunk"
    id = Column(Integer, primary_key=True, autoincrement=True)
    kb_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=False)
    content = Column(Text, nullable=False, comment="文档切片文本")
    embedding = Column(Vector(384), comment="文本向量，维度384（all-MiniLM-L6-v2）")
    source_name = Column(String(255), comment="原文件名称")
    created_at = Column(DateTime, server_default=func.now())
