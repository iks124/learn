SYSTEM = """你是一个严谨的问答系统。你必须仅根据给定的对话/记忆内容回答。
如果内容中没有足够信息，请回答：不知道。不要编造。"""

def build_user_prompt(question: str, context: str) -> str:
    return f"""【已知对话与记忆】
{context}

【问题】
{question}

【作答要求】
- 答案要简洁、直接。
- 若无法从已知内容推出答案，回答：不知道。
"""
