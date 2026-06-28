from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from src.core.settings import settings
from src.core.logger import log
from sqlalchemy import text

# --------------------------知识点分割线---------------------------
# 1. asyncpg 异步驱动：连接串前缀 postgresql+asyncpg://
# 2. 连接池参数 pool_size：常驻连接数，max_overflow：突发扩容连接
# 3. declarative_base：所有数据表模型的父类，统一ORM基类
# 4. async_sessionmaker：生产异步会话，每次数据库操作开启/关闭会话
# ----------------------------------------------------------------

# 拼接异步数据库连接字符串
DB_URL = (
    f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
    f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
)

# 创建异步引擎
async_engine = create_async_engine(
    DB_URL,
    pool_size=10,          # 常驻连接池10个连接
    max_overflow=20,       # 峰值最多额外扩容20个
    pool_recycle=1800,     # 30分钟回收闲置连接，避免断开
    echo=settings.ENV == "dev"  # 开发环境打印SQL日志，生产关闭
)

# 异步会话工厂
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False
)

# ORM 模型基类
Base = declarative_base()

# 依赖注入：获取数据库会话
async def get_db_session() -> AsyncSession:
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception as e:
        log.error(f"数据库会话异常: {str(e)}", exc_info=True)
        await session.rollback()
        raise
    finally:
        await session.close()

# 全局初始化：创建所有数据表（服务启动执行）
async def init_db_tables():
    async with async_engine.begin() as conn:
        # 开启向量拓展
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # 自动扫描所有继承Base的模型，不存在则创建表
        await conn.run_sync(Base.metadata.create_all)
    log.info("数据库数据表初始化完成")

