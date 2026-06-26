# AI Agent 企业知识库对话底座
基于Python + FastAPI + Async SQLAlchemy + 通义千问兼容接口开发的轻量化AI Agent后端底座，支持流式聊天、历史持久化、RAG提示模板、自动上下文截断，可拓展工具调用、向量检索。

## 技术栈
- 运行时：Python3.14
- Web框架：FastAPI 异步高并发
- HTTP客户端：httpx.AsyncClient
- 数据库：PostgreSQL 15（asyncpg异步驱动）
- 依赖管理：uv
- 容器化：Docker + Docker Compose
- 日志：Loguru 分层可观测日志
- 数据校验：Pydantic v2
- Token处理：tiktoken 精准计数滑动窗口截断

## 核心功能
1. 兼容OpenAI格式大模型调用，一次性问答 + SSE流式打字机输出
2. 全链路日志 + RequestID 追踪，线上问题快速排查
3. 自动Token滑动窗口截断，解决大模型上下文超限、幻觉问题
4. 4套生产级Prompt模板：知识库RAG、CoT逻辑推理、ReAct工具调用、结构化数据提取
5. 异步PostgreSQL持久化会话与聊天记录，重启服务不丢失历史
6. 简易Header鉴权机制，可无缝升级JWT/OAuth2
7. Docker一键部署，后端+数据库自动编排，数据持久化不丢失

## 本地开发启动
1. 配置.env文件
```env
# 服务
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
ENV=dev

# LLM
LLM_API_KEY=你的通义千问key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-turbo

# PostgreSQL
DB_USER=agent
DB_PASSWORD=123456
DB_NAME=agent_db
DB_HOST=localhost
DB_PORT=5432

2. 完整安装依赖命令
# 1. 清理旧虚拟环境（环境异常时执行，可选）
rm -rf .venv

# 2. 安装项目全部开发依赖
uv sync --extra dev

# 若缺失异步数据库驱动，单独补充安装
uv add asyncpg "sqlalchemy[asyncio]"

3. 构建镜像并后台启动全部服务
# 启动
docker compose up -d --build
# 测试
uv run python test_day6.py



