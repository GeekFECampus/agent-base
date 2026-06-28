from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from src.api.schema import ChatRequest, ChatStreamResponse
from src.api.deps import get_llm_client, TokenDep, RequestIdDep, DBSessionDep
from src.core.logger import log
from src.db import crud
from src.utils.rag.rag_chain import build_rag_prompt
from src.utils.prompt_manager import prompt_manager
import json

# --------------------------知识点分割线---------------------------
# 知识点1：StreamingResponse SSE流式响应
# text/event-stream 是前端聊天流式输出标准协议，相比WebSocket轻量化，无状态
# 生成器yield逐段输出，实现打字机效果
# 知识点2：路由分层
# 拆分chat、admin、knowledge多个router，大型项目代码不拥挤
# ----------------------------------------------------------------

router = APIRouter(prefix="/chat", tags=["对话接口"])

@router.post("/completion")
async def chat_completion(
    req: ChatRequest,
    kb_id: int, # 新增可选知识库ID
    token: TokenDep,
    request_id: RequestIdDep,
    db: DBSessionDep,
    llm = Depends(get_llm_client)
):
    log.info(f"[{request_id}] 用户提问: {req.prompt}, kb_id={kb_id}")
    # 1. 用户校验，不存在则自动创建用户
    user = await crud.get_user_by_token(db, token)
    if not user:
        user = await crud.create_user(db, token)
    # 2. 创建默认会话（简易版，正式可前端传session_id）
    session = await crud.create_chat_session(db, user.id)
    # 3. 存储用户提问
    await crud.add_chat_message(db, session.id, "user", req.prompt)
    if kb_id:
        # 有知识库ID，走RAG检索
        system_prompt = await build_rag_prompt(db, kb_id, req.prompt, req.history)
    else:
        # 无知识库，使用通用问答模板
        system_prompt = prompt_manager.build_normal_chat_prompt(req.prompt, req.history)
    # 4. 调用大模型
    reply = await llm.chat(req.prompt, req.history, system_prompt=system_prompt)
    # 5. 存储助手回答
    await crud.add_chat_message(db, session.id, "assistant", reply)
    return {"request_id": request_id, "session_id": session.id, "reply": reply}

@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    kb_id: int, # 新增可选知识库ID
    token: TokenDep,
    request_id: RequestIdDep,
    db: DBSessionDep,
    llm = Depends(get_llm_client)
):
    """SSE流式对话接口，前端实时打字输出"""
    log.info(f"[{request_id}] 开启流式对话: {req.prompt}, kb_id={kb_id}")
    # 1. 用户校验，不存在则自动创建用户
    user = await crud.get_user_by_token(db, token)
    if not user:
        user = await crud.create_user(db, token)
    # 2. 创建默认会话（简易版，正式可前端传session_id）
    session = await crud.create_chat_session(db, user.id)
    # 3. 存储用户提问
    await crud.add_chat_message(db, session.id, "user", req.prompt)
    if kb_id:
        # 有知识库ID，走RAG检索
        system_prompt = await build_rag_prompt(db, kb_id, req.prompt, req.history)
    else:
        # 无知识库，使用通用问答模板
        system_prompt = prompt_manager.build_normal_chat_prompt(req.prompt, req.history, system_prompt=system_prompt)
    async def stream_generator():
        async for chunk_text in llm.stream_chat(req.prompt, req.history):
            # 按照SSE标准封装data内容
            resp_data = ChatStreamResponse(content=chunk_text, finish=False).model_dump_json()
            yield f"data: {resp_data}\n\n"
        # 流结束标识
        end_data = ChatStreamResponse(content="", finish=True).model_dump_json()
        # yield f"data: {end_data}\n\n"
        # 5. 存储助手回答
        await crud.add_chat_message(db, session.id, "assistant", end_data)
        yield f"data: {end_data}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"X-Request-Id": request_id}
    )
