from fastapi import FastAPI, Request
from src.api.chat_router import router as chat_routers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.core.logger import log
from src.api.history_router import router as history_router
from src.api.rag_router import router as kb_router
from src.core.db import init_db_tables
import time

# --------------------------知识点分割线---------------------------
# 知识点1：FastAPI生命周期事件
# startup服务启动前初始化资源，shutdown关闭时释放http/数据库连接
# 知识点2：全局中间件
# 拦截所有请求，记录请求耗时、url、状态码，全局统一日志
# ----------------------------------------------------------------

app = FastAPI(title="Agent-Base 企业AI智能体底座", version="0.1.0")

# 注册路由
app.include_router(chat_routers)
app.include_router(history_router)
app.include_router(kb_router)

# 全局请求中间件
@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    cost = round((time.time() - start) * 1000, 2)
    log.info(f"PATH:{request.url.path} | STATUS:{response.status_code} | COST:{cost}ms")
    return response

# 服务启动钩子
@app.on_event("startup")
async def startup_event():
    await init_db_tables()
    log.info("==== Agent-Base 服务启动完成 ====")

# 服务关闭钩子
@app.on_event("shutdown")
async def shutdown_event():
    log.info("==== Agent-Base 服务正在关闭 ====")

# startup 钩子初始化数据表
# @app.on_event("startup")
# async def startup_event():
#     await init_db_tables()
#     log.info("==== Agent-Base 服务启动完成 ====")

# 全局异常捕获
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"全局接口异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "msg": "服务内部异常，请查看日志排查", "detail": str(exc)}
    )

# 跨域配置，前端对接必备
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    from src.core.settings import settings
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.ENV == "dev" # 开发环境热重载
    )
