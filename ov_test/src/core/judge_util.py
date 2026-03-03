import json


def llm_grader(
    llm_client,
    model: str,
    question: str,
    gold_answer: str,
    response: str,
    dataset_name: str = "Locomo"
) -> bool:
    """
    兼容原 ruc-ov-eval 的 judge 逻辑，但不再依赖 langchain。
    llm_client 期望支持：
      - 推荐：llm_client.generate(prompt: str) -> str
      - 兜底：如果传的是 openai client（OpenAI SDK），也可在这里扩展（当前以 generate 为主）

    返回 bool: True 表示 CORRECT, False 表示 WRONG
    """

    # 1) 根据 dataset_name 路由选择 Prompt（保持你原逻辑）
    if "locomo" in (dataset_name or "").lower():
        system_prompt = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """
        ACCURACY_PROMPT = f"""
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
    """
    else:
        system_prompt = """
        You are an expert grader that determines if an AI-generated answer matches the gold standard (ground truth) answer for a given question.
        """
        ACCURACY_PROMPT = f"""
        Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
            (1) A question
            (2) A 'gold' (ground truth) answer
            (3) A generated answer

        Grading rules:
        - If the generated answer correctly encompasses the core semantic meaning or facts of the gold answer, grade it as CORRECT.
        - If the generated answer contradicts the gold answer or misses the key factual information, it is WRONG.

        Question: {question}
        Gold answer: {gold_answer}
        Generated answer: {response}

        First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
        Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
        """

    # 2) 构造“单字符串 prompt”，替代 langchain messages（行为等价）
    prompt = (
        f"{system_prompt.strip()}\n\n"
        f"{ACCURACY_PROMPT.strip()}\n"
    )

    # 3) 调用 LLM（优先走你自己的 wrapper：generate）
    content = None
    try:
        if hasattr(llm_client, "generate"):
            content = llm_client.generate(prompt)
        else:
            # 兜底：如果用户传进来的是一个可调用对象
            content = llm_client(prompt)
    except Exception as e:
        # Judge 调用失败，按不正确处理（也可以改成 raise）
        return False

    if content is None:
        return False

    # 4) 解析结果：保持你原逻辑
    try:
        result = json.loads(content)
        label = result.get("is_correct", result.get("label", "WRONG"))
        return str(label).strip().lower() == "correct"
    except json.JSONDecodeError:
        # 容错：防止 LLM 没按格式输出 JSON
        return "CORRECT" in str(content).upper()