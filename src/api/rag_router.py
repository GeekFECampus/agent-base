from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import tempfile
import os

from src.core.db import AsyncSessionLocal
from src.db import crud as crud_kb
from src.utils.rag.chunker import split_text, load_pdf_text, load_txt_text
from src.utils.rag.embedder import batch_get_embedding
from src.utils.rag.retriever import format_chunk_context
from src.utils.rag.rag_chain import build_rag_prompt
from src.api.chat_router import get_llm_client
from src.core.base import BaseLLM

router = APIRouter(prefix="/kb", tags=["知识库RAG管理"])

def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1. 创建知识库
@router.post("/create")
async def create_knowledge_base(name: str, desc: str = "", db: AsyncSession = Depends(get_db)):
    kb = await crud_kb.create_kb(db, name, desc)
    return {"code": 0, "msg": "知识库创建成功", "data": {"kb_id": kb.id, "name": kb.name}}

# 2. 查询所有知识库
@router.get("/list")
async def list_kb(db: AsyncSession = Depends(get_db)):
    kb_list = await crud_kb.list_all_kb(db)
    return {"code": 0, "data": [{"id": item.id, "name": item.name, "desc": item.desc} for item in kb_list]}

# 3. 上传PDF/TXT文档，自动切片+向量化入库
@router.post("/upload/{kb_id}")
async def upload_document(
    kb_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    kb = await crud_kb.get_kb_by_id(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    suffix = file.filename.split(".")[-1].lower()
    full_text = ""
    # 临时存储文件解析
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        if suffix == "pdf":
            full_text = load_pdf_text(tmp_path)
        elif suffix == "txt":
            full_text = load_txt_text(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="仅支持pdf、txt格式文件")
    finally:
        os.unlink(tmp_path)

    if not full_text.strip():
        raise HTTPException(status_code=400, detail="文档无有效文本内容")

    # 文本切片
    chunks = split_text(full_text, chunk_size=500, overlap=100)
    # 批量生成向量
    vecs = batch_get_embedding(chunks)
    # 组装批量数据入库
    batch_data = [
        {
            "kb_id": kb_id,
            "content": chunk,
            "embedding": vec,
            "source_name": file.filename
        }
        for chunk, vec in zip(chunks, vecs)
    ]
    await crud_kb.batch_insert_chunks(db, batch_data)
    return {"code": 0, "msg": f"文档上传完成，共生成{len(chunks)}条知识库切片"}

# 4. 基于知识库问答
@router.post("/chat/{kb_id}")
async def kb_chat(
    kb_id: int,
    query: str,
    db: AsyncSession = Depends(get_db),
    llm: BaseLLM = Depends(get_llm_client)
):
    kb = await crud_kb.get_kb_by_id(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    # 生成RAG系统提示词
    sys_prompt = await build_rag_prompt(db, kb_id, query)
    resp = await llm.chat(prompt=query, system_prompt=sys_prompt, history=[])
    return {"code": 0, "answer": resp}
