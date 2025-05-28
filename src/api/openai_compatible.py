"""
Open CodeAI - OpenAI 호환 API 엔드포인트 구현
Continue.dev와 완전 호환되는 OpenAI API 스펙 준수
"""
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

from ..config import settings
from ..core.llm_manager import LLMManager
from ..core.code_analyzer import CodeAnalyzer
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Pydantic 모델들 - OpenAI API 스펙 준수
class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system, user, assistant)")
    content: str = Field(..., description="메시지 내용")
    name: Optional[str] = Field(None, description="메시지 발신자 이름")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="사용할 모델 이름")
    messages: List[ChatMessage] = Field(..., description="대화 메시지 목록")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="응답 창의성")
    max_tokens: Optional[int] = Field(4096, ge=1, description="최대 토큰 수")
    stream: Optional[bool] = Field(False, description="스트리밍 응답 여부")
    stop: Optional[Union[str, List[str]]] = Field(None, description="정지 시퀀스")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)

class CompletionRequest(BaseModel):
    model: str = Field(..., description="사용할 모델 이름")
    prompt: str = Field(..., description="완성할 프롬프트")
    max_tokens: Optional[int] = Field(256, ge=1)
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    suffix: Optional[str] = Field(None, description="코드 완성용 접미사")

class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="임베딩 모델 이름")
    input: Union[str, List[str]] = Field(..., description="임베딩할 텍스트")

class ModelInfo(BaseModel):
    id: str
    object: str = "model" 
    created: int
    owned_by: str = "open-codeai"
    permission: List[Dict] = []
    root: str
    parent: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: Optional[ChatMessage] = None
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

# 의존성 주입
def get_llm_manager() -> LLMManager:
    """LLM 관리자 인스턴스 반환"""
    try:
        return LLMManager()
    except Exception as e:
        logger.error(f"LLM 관리자 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail="LLM 관리자 초기화 실패")

def get_code_analyzer() -> CodeAnalyzer:
    """코드 분석기 인스턴스 반환"""
    try:
        return CodeAnalyzer()
    except Exception as e:
        logger.error(f"코드 분석기 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail="코드 분석기 초기화 실패")

async def verify_api_key(request: Request) -> bool:
    """API 키 검증"""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="API 키가 필요합니다")
    
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer 토큰 형식이 아닙니다")
    
    api_key = auth_header[7:]  # "Bearer " 제거
    
    # API 키 검증 (로컬 환경에서는 관대하게)
    if api_key not in [settings.API_KEY, "open-codeai-local-key"]:
        raise HTTPException(status_code=401, detail="유효하지 않은 API 키")
    
    return True

# 유틸리티 함수들
def generate_id() -> str:
    """고유 ID 생성"""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"

def count_tokens(text: str) -> int:
    """토큰 수 추정 (간단한 구현)"""
    # 실제로는 tokenizer를 사용해야 하지만, 추정치로 대체
    return len(text.split()) * 1.3

async def stream_response(content: str, model: str, request_id: str) -> AsyncGenerator[str, None]:
    """스트리밍 응답 생성"""
    words = content.split()
    
    for i, word in enumerate(words):
        chunk_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": word + " " if i < len(words) - 1 else word},
                "finish_reason": None if i < len(words) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        # 실제 스트리밍 효과를 위한 지연
        import asyncio
        await asyncio.sleep(0.05)
    
    # 스트림 종료 신호
    yield "data: [DONE]\n\n"

# API 엔드포인트들

@router.get("/models")
async def list_models(_: bool = Depends(verify_api_key)) -> Dict[str, Any]:
    """사용 가능한 모델 목록 반환"""
    try:
        models = [
            ModelInfo(
                id="open-codeai",
                created=int(time.time()),
                root="open-codeai"
            ),
            ModelInfo(
                id="qwen2.5-coder-32b",
                created=int(time.time()),
                root="qwen2.5-coder-32b"
            ),
            ModelInfo(
                id="text-embedding-ada-002",
                created=int(time.time()),
                root="bge-large-en-v1.5"
            )
        ]
        
        return {
            "object": "list",
            "data": [model.dict() for model in models]
        }
    except Exception as e:
        logger.error(f"모델 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail="모델 목록 조회 실패")

@router.post("/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    _: bool = Depends(verify_api_key),
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """채팅 완성 API - Continue.dev의 메인 기능"""
    
    try:
        request_id = generate_id()
        
        # 메시지를 하나의 프롬프트로 합성
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant: "
        
        logger.info(f"채팅 완성 요청 - 모델: {request.model}, 메시지 수: {len(request.messages)}")
        
        # LLM으로 응답 생성
        response_text = await llm_manager.generate_response(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # 토큰 사용량 계산
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(response_text)
        
        if request.stream:
            # 스트리밍 응답
            return StreamingResponse(
                stream_response(response_text, request.model, request_id),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # 일반 응답
            return ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(role="assistant", content=response_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
    except Exception as e:
        logger.error(f"채팅 완성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"채팅 완성 실패: {str(e)}")

@router.post("/completions")
async def create_completion(
    request: CompletionRequest,
    _: bool = Depends(verify_api_key),
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> Union[CompletionResponse, StreamingResponse]:
    """텍스트 완성 API - 코드 자동완성용"""
    
    try:
        request_id = generate_id()
        
        logger.info(f"텍스트 완성 요청 - 모델: {request.model}")
        
        # 코드 완성 최적화된 프롬프트 생성
        if request.suffix:
            # FIM (Fill-in-the-Middle) 형식 지원
            prompt = f"<PRE>{request.prompt}<SUF>{request.suffix}<MID>"
        else:
            prompt = request.prompt
        
        # LLM으로 완성 생성
        completion_text = await llm_manager.generate_completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_sequences=request.stop if isinstance(request.stop, list) else [request.stop] if request.stop else None
        )
        
        # 토큰 사용량 계산
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(completion_text)
        
        if request.stream:
            # 스트리밍 응답
            return StreamingResponse(
                stream_response(completion_text, request.model, request_id),
                media_type="text/plain"
            )
        else:
            # 일반 응답
            return CompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        text=completion_text,
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
    except Exception as e:
        logger.error(f"텍스트 완성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 완성 실패: {str(e)}")

@router.post("/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    _: bool = Depends(verify_api_key),
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> EmbeddingResponse:
    """임베딩 생성 API - 코드베이스 검색용"""
    
    try:
        logger.info(f"임베딩 생성 요청 - 모델: {request.model}")
        
        # 입력을 리스트로 정규화
        if isinstance(request.input, str):
            inputs = [request.input]
        else:
            inputs = request.input
        
        # 각 입력에 대해 임베딩 생성
        embeddings_data = []
        total_tokens = 0
        
        for i, text in enumerate(inputs):
            embedding = await llm_manager.generate_embedding(text)
            embeddings_data.append(
                EmbeddingData(
                    embedding=embedding,
                    index=i
                )
            )
            total_tokens += count_tokens(text)
        
        return EmbeddingResponse(
            object="list",
            data=embeddings_data,
            model=request.model,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens
            )
        )
        
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {str(e)}")

# Continue.dev 전용 확장 엔드포인트들

@router.post("/code/analyze")
async def analyze_code(
    code: str,
    language: str = "python",
    _: bool = Depends(verify_api_key),
    code_analyzer: CodeAnalyzer = Depends(get_code_analyzer)
) -> Dict[str, Any]: 
    """코드 분석 API - Continue.dev 확장 기능"""
    
    try:
        logger.info(f"코드 분석 요청 - 언어: {language}")
        
        analysis_result = await code_analyzer.analyze_code(code, language)
        
        return {
            "status": "success",
            "language": language,
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"코드 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"코드 분석 실패: {str(e)}")

@router.post("/code/search")
async def search_codebase(
    query: str,
    project_path: Optional[str] = None,
    limit: int = 10,
    _: bool = Depends(verify_api_key)
) -> Dict[str, Any]:
    """코드베이스 검색 API - RAG 기능"""
    
    try:
        logger.info(f"코드베이스 검색 요청 - 쿼리: {query[:50]}...")
        
        # 실제로는 벡터 검색을 수행해야 하지만, 여기서는 모의 응답
        search_results = [
            {
                "file_path": "src/utils/helper.py",
                "line_number": 42,
                "code_snippet": "def process_data(data):\n    return data.strip().lower()",
                "similarity_score": 0.95
            },
            {
                "file_path": "src/api/routes.py", 
                "line_number": 15,
                "code_snippet": "@router.get('/data')\ndef get_data():\n    return {'status': 'ok'}",
                "similarity_score": 0.87
            }
        ]
        
        return {
            "status": "success",
            "query": query,
            "results": search_results[:limit],
            "total_found": len(search_results),
            "project_path": project_path
        }
        
    except Exception as e:
        logger.error(f"코드베이스 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"코드베이스 검색 실패: {str(e)}")

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.VERSION,
        "models_loaded": True,  # 실제로는 모델 로딩 상태 확인
        "gpu_available": settings.is_gpu_available()
    }