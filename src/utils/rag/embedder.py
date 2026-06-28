from sentence_transformers import SentenceTransformer
from typing import List

# 全局加载模型，只初始化一次
model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

def get_text_embedding(text: str) -> List[float]:
    """单段文本生成384维向量"""
    vec = model.encode(text)
    return vec.tolist()

def batch_get_embedding(text_list: list[str]) -> List[List[float]]:
    """批量生成向量"""
    vecs = model.encode(text_list)
    return [v.tolist() for v in vecs]
