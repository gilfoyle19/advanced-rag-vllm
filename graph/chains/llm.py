from langchain_openai import ChatOpenAI

from graph.config import VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL


def get_llm(max_tokens: int = 1024) -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        temperature=0.1,
        max_tokens=max_tokens,
        max_retries=3,
    )
