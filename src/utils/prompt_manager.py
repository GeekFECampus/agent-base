from typing import List, Optional, Dict, Any
from src.api.schema import ChatHistoryItem
from src.utils.token_utils import token_counter
from src.core.logger import log

# --------------------------知识点分割线---------------------------
# 1. 模板解耦：所有Prompt字符串统一管理，业务层只传参数，不用硬编码长文本
# 2. 自动Token校验：拼接完成后自动检测是否接近窗口上限，提前告警
# 3. 模板分层：通用问答、CoT推理、ReAct工具调用、结构化提取四大生产模板
# 4. 兼容多轮历史、外部知识库上下文、工具描述入参
# ----------------------------------------------------------------

class PromptManager:
    """全局提示词模板管理单例"""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            log.info("PromptManager 提示模板管理器初始化完成")
        return cls._instance

    # ===================== 模板1：通用知识库问答模板（RAG专用） =====================
    def build_rag_chat_prompt(self, user_query: str, context_docs: str, history: list) -> str:
        template = """
    你是企业专属知识库问答助手，回答必须严格遵守以下规则：
    1. 只能依据下方【参考知识库资料】作答，绝对禁止编造不存在的内容；
    2. 如果参考资料没有相关信息，直接回复：当前知识库未查询到相关内容，不要自行拓展知识；
    3. 回答条理清晰，分点输出，不要输出多余闲聊内容；
    4. 不要输出参考资料原文，用自己语言总结。

    【参考知识库资料】
    {context_docs}

    用户当前问题：{user_query}
    历史对话记录：{history}
    """
        return template.format(
            context_docs=context_docs,
            user_query=user_query,
            history=history
        )

    # ===================== 模板2：CoT思维链复杂推理模板 =====================
    def build_cot_reason_prompt(self, user_query: str, history: Optional[List[ChatHistoryItem]] = None) -> str:
        """复杂逻辑、计算、业务分析专用，强制分步思考"""
        system_prompt = """
你是专业逻辑分析助手，回答必须遵循CoT思维链格式：
1. 第一步：拆解用户问题，列出需要分析的关键点；
2. 第二步：逐点分步推导、计算、验证；
3. 第三步：汇总所有推导结论，给出最终明确答案。
禁止直接给出最终结果，必须展示完整思考过程。
"""
        user_part = f"""
历史对话记录：
{self._format_history(history)}

待分析问题：{user_query}
"""
        full_prompt = system_prompt + user_part
        self._check_prompt_token(full_prompt, threshold=3800)
        return full_prompt

    # ===================== 模板3：ReAct工具调用模板（Agent核心） =====================
    def build_react_tool_prompt(self, user_query: str, tool_list: List[Dict[str, Any]], history: Optional[List[ChatHistoryItem]] = None) -> str:
        """
        ReAct推理行动模板，用于自动选择工具、调用工具
        :param tool_list: 可用工具描述数组（name/desc/params）
        """
        tool_desc_text = self._format_tool_list(tool_list)
        system_prompt = f"""
你是具备工具调用能力的AI智能体，遵循ReAct循环逻辑完成任务：
循环三步：Thought(思考) → Action(工具调用) → Observation(工具返回结果)
规则：
1. 先判断当前问题是否需要调用工具，仅依靠自身知识无法回答时才选择工具；
2. 工具调用输出严格JSON格式，禁止自由文本；
3. 工具列表如下：
{tool_desc_text}
4. 若无合适工具可解决问题，直接输出最终答案，无需调用工具。

输出格式规范：
1. 需要调用工具时输出：
{{
  "thought": "思考我需要用哪个工具，为什么",
  "action": "工具名称",
  "params": {{对应工具入参键值对}}
}}
2. 无需工具直接回答时输出：
{{
  "thought": "无需调用工具，依靠已有信息直接回答",
  "answer": "完整回答文本"
}}
"""
        user_part = f"""
历史对话：
{self._format_history(history)}

用户任务：{user_query}
"""
        full_prompt = system_prompt + user_part
        self._check_prompt_token(full_prompt, threshold=3200)
        return full_prompt

    # ===================== 模板4：结构化JSON提取模板 =====================
    def build_struct_extract_prompt(self, user_text: str, extract_schema: str) -> str:
        """
        文本信息提取，强制输出固定JSON结构
        :param user_text: 需要提取信息的原始文本
        :param extract_schema: 要求输出的JSON字段定义
        """
        system_prompt = f"""
你是结构化数据提取助手，严格按照要求从原文抽取信息：
1. 只输出标准JSON，无任何多余文字、解释、注释；
2. 原文无对应字段内容时填空字符串""；
3. 输出字段规范：
{extract_schema}
"""
        user_part = f"待提取原文：{user_text}"
        full_prompt = system_prompt + user_part
        self._check_prompt_token(full_prompt, threshold=4000)
        return full_prompt

    # ===================== 内部通用工具方法 =====================
    def _format_history(self, history: Optional[List[ChatHistoryItem]]) -> str:
        """格式化多轮对话历史为文本字符串"""
        if not history:
            return "无历史对话"
        lines = []
        for msg in history:
            role_name = "用户" if msg.role == "user" else "助手"
            lines.append(f"{role_name}：{msg.content}")
        return "\n".join(lines)

    def _format_tool_list(self, tool_list: List[Dict[str, Any]]) -> str:
        """格式化工具列表为可读文本"""
        lines = []
        for idx, tool in enumerate(tool_list):
            lines.append(f"工具{idx+1}：名称={tool['name']}，功能描述={tool['desc']}，入参={tool['params']}")
        return "\n".join(lines)

    def _check_prompt_token(self, prompt_text: str, threshold: int):
        """自动校验Prompt总token，临近上限打印告警日志"""
        token_num = token_counter.count_single_text(prompt_text)
        if token_num >= threshold:
            log.warning(f"Prompt token 接近阈值！当前token：{token_num}，阈值：{threshold}，存在上下文超限风险")
        else:
            log.debug(f"Prompt token校验通过，总长度：{token_num}")
# 全局单例导出，项目任意位置直接调用
prompt_manager = PromptManager()
