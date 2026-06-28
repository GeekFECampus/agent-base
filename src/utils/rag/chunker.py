def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    固定长度切片 + 重叠窗口，避免上下文断裂
    :param text: 原始全文
    :param chunk_size: 单块字符上限
    :param overlap: 块之间重叠字符数
    :return: 切片文本列表
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        # 滑动窗口，带上重叠
        start = end - overlap
    return chunks

# PDF读取工具
from PyPDF2 import PdfReader

def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def load_txt_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
