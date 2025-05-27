from typing import Any, Dict

def chat_completion(model: str, messages: list, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
    # 실제론 LLM 서버에 요청
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "(모의 LLM 응답)"},
            "finish_reason": "stop"
        }],
        "model": model
    }

def completion(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "choices": [{
            "index": 0,
            "text": "(모의 LLM 자동완성)",
            "finish_reason": "stop"
        }],
        "model": model
    } 