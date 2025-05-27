from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

router = APIRouter()

# 요청/응답 모델 정의
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    choices: Any
    model: str

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class CompletionResponse(BaseModel):
    id: str
    object: str
    choices: Any
    model: str

# 실제 LLM 호출은 core.llm_server에서 import (여기선 mock)
def mock_chat_completion(request: ChatCompletionRequest):
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "(모의 응답) 코드 리뷰 결과입니다."},
            "finish_reason": "stop"
        }],
        "model": request.model
    }

def mock_completion(request: CompletionRequest):
    return {
        "id": "cmpl-mock",
        "object": "text_completion",
        "choices": [{
            "index": 0,
            "text": "(모의 응답) 자동완성 결과입니다.",
            "finish_reason": "stop"
        }],
        "model": request.model
    }

@router.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    # 실제론 core.llm_server.chat_completion 호출
    return mock_chat_completion(request)

@router.post("/completions")
def completions(request: CompletionRequest):
    return mock_completion(request)

@router.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "open-codeai", "object": "model", "owned_by": "open-codeai"}
        ]
    } 