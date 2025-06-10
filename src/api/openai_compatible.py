"""
Open CodeAI - 업데이트된 OpenAI 호환 API
Function Calling과 RAG 시스템이 통합된 API
"""
import time
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from datetime import datetime
import re

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import json

from src.config import settings
from src.core.llm_manager import get_llm_manager
from src.core.function_calling import get_function_registry, EnhancedLLMManager
from src.core.rag_system import get_rag_system
from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Pydantic 모델들 - OpenAI API 스펙 확장
class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system, user, assistant, function)")
    content: Optional[str] = Field(None, description="메시지 내용")
    name: Optional[str] = Field(None, description="메시지 발신자 이름")
    function_call: Optional[Dict[str, Any]] = Field(None, description="함수 호출 정보")

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="사용할 모델 이름")
    messages: List[ChatMessage] = Field(..., description="대화 메시지 목록")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="응답 창의성")
    max_tokens: Optional[int] = Field(4096, ge=1, description="최대 토큰 수")
    stream: Optional[bool] = Field(False, description="스트리밍 응답 여부")
    functions: Optional[List[FunctionDefinition]] = Field(None, description="사용 가능한 함수들")
    function_call: Optional[Union[str, Dict[str, str]]] = Field(None, description="함수 호출 설정")
    stop: Optional[Union[str, List[str]]] = Field(None, description="정지 시퀀스")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(1, ge=1, le=10)
    # Open CodeAI 확장 파라미터
    enable_rag: Optional[bool] = Field(True, description="RAG 시스템 사용 여부")
    enable_function_calling: Optional[bool] = Field(True, description="함수 호출 사용 여부")

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

# Open CodeAI 전용 요청 모델들
class CodeSearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    max_results: Optional[int] = Field(10, ge=1, le=50)
    file_type: Optional[str] = Field(None)
    project_path: Optional[str] = Field(None)

class IndexProjectRequest(BaseModel):
    project_path: str = Field(..., description="인덱싱할 프로젝트 경로")
    max_files: Optional[int] = Field(1000, ge=1, le=10000)
    force_reindex: Optional[bool] = Field(False)

# 응답 모델들
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "open-codeai"
    permission: List[Dict] = Field(default_factory=list)
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

# 의존성 주입
def get_enhanced_llm_manager() -> EnhancedLLMManager:
    """Enhanced LLM 관리자 인스턴스 반환"""
    try:
        base_llm = get_llm_manager()
        return EnhancedLLMManager(base_llm)
    except Exception as e:
        logger.error(f"Enhanced LLM 관리자 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail="LLM 관리자 초기화 실패")

# 유틸리티 함수들
def generate_id() -> str:
    """고유 ID 생성"""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"

def count_tokens(text: str) -> int:
    """토큰 수 추정"""
    return int(len(text.split()) * 1.3)

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
        
        import asyncio
        await asyncio.sleep(0.05)
    
    yield "data: [DONE]\n\n"

def format_for_continue_extension(code_content: str, language: str = None, file_hint: str = None) -> str:
    """Continue 확장을 위한 코드 블록 포맷팅
    
    Args:
        code_content: 코드 내용
        language: 코드 언어 (예: python, javascript)
        file_hint: 파일 경로 힌트 (예: src/main.py)
    
    Returns:
        Continue 확장이 인식할 수 있는 형식의 코드 블록
    """
    # 언어가 없는 경우 텍스트로 처리
    if not language:
        language = "text"
    
    # 파일 경로 정규화
    if file_hint:
        # /workspace/, ./, ../ 등의 경로 prefix 제거
        file_hint = file_hint.replace("/workspace/", "").replace("./", "").replace("../", "")
        # 절대 경로를 상대 경로로 변환
        if file_hint.startswith("/"):
            file_hint = file_hint[1:]
    
    # IntelliJ와 VSCode 모두에서 인식되도록 정확한 포맷팅
    # 1. 언어와 파일 경로 사이에 정확히 한 칸의 공백
    # 2. 파일 경로가 없는 경우 공백 없음
    # 3. 코드 블록 시작과 끝에 정확히 세 개의 백틱
    return f"```{language}{(' ' + file_hint) if file_hint else ''}\n{code_content}\n```"

def extract_explanation_and_codeblocks(response: str) -> str:
    """
    LLM 응답에서 설명+코드블록 쌍만 추출 (불필요한 안내/적용방법 등 제거)
    여러 쌍이 있으면 모두 반환
    """
    codeblock_pattern = r'(```[\w\s./-]*\n.*?```)'  # 코드블록 전체
    parts = re.split(codeblock_pattern, response, flags=re.DOTALL)
    result = []
    for i in range(1, len(parts), 2):
        explanation = parts[i-1].strip()
        codeblock = parts[i].strip()
        # 불필요한 안내/적용방법/추가설명 등은 설명에서 제거
        if explanation:
            if any(x in explanation.lower() for x in ["적용 방법", "apply button", "agent mode", "mode selector", "추가적인 도움이 필요하시면"]):
                explanation = ""
        if explanation:
            result.append(explanation)
        result.append(codeblock)
    return "\n\n".join(result) if result else response

# API 엔드포인트들

@router.get("/models")
async def list_models() -> Dict[str, Any]:
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

@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    enhanced_llm: EnhancedLLMManager = Depends(get_enhanced_llm_manager)
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """채팅 완성 API - Function Calling과 RAG 통합"""
    
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
            elif msg.role == "function":
                prompt += f"Function Result: {msg.content}\n\n"
        
        prompt += "Assistant: "
        
        logger.info(f"채팅 완성 요청 - 모델: {request.model}, 메시지 수: {len(request.messages)}")
        
        # Enhanced LLM으로 응답 생성 (Function Calling과 RAG 포함)
        response_text = await enhanced_llm.generate_response_with_functions(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            enable_function_calling=request.enable_function_calling
        )
        response_text = format_for_continue_extension(response_text)
        response_text = extract_explanation_and_codeblocks(response_text)
        
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

@router.post("/completions", response_model=None)
async def create_completion(
    request: CompletionRequest,
    llm_manager = Depends(get_llm_manager)
) -> Union[Dict[str, Any], StreamingResponse]:
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
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "text": completion_text,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            
    except Exception as e:
        logger.error(f"텍스트 완성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"텍스트 완성 실패: {str(e)}")

@router.post("/embeddings")
async def create_embeddings(
    request: EmbeddingRequest,
    llm_manager = Depends(get_llm_manager)
) -> Dict[str, Any]:
    """임베딩 생성 API"""
    
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
            embeddings_data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
            total_tokens += count_tokens(text)
        
        return {
            "object": "list",
            "data": embeddings_data,
            "model": request.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "completion_tokens": 0,
                "total_tokens": total_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {str(e)}")

# Open CodeAI 전용 확장 엔드포인트들

@router.post("/code/search")
async def search_codebase(
    request: CodeSearchRequest,
    rag_system = Depends(get_rag_system)
) -> Dict[str, Any]:
    """코드베이스 검색 API"""
    
    try:
        logger.info(f"코드베이스 검색 요청 - 쿼리: {request.query[:50]}...")
        
        search_results = await rag_system.search_codebase(
            query=request.query,
            project_path=request.project_path,
            k=request.max_results
        )
        
        formatted_results = []
        for result in search_results:
            chunk = result.chunk
            formatted_result = {
                "file_path": chunk.file_path,
                "content": chunk.content,
                "line_range": f"{chunk.start_line}-{chunk.end_line}",
                "chunk_type": chunk.chunk_type,
                "language": chunk.language,
                "similarity_score": result.similarity_score,
                "relevance_score": result.relevance_score,
                "metadata": chunk.metadata or {}
            }
            
            if result.graph_connections:
                formatted_result["related_files"] = result.graph_connections
            
            formatted_results.append(formatted_result)
        
        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "total_found": len(formatted_results),
            "project_path": request.project_path
        }
        
    except Exception as e:
        logger.error(f"코드베이스 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"코드베이스 검색 실패: {str(e)}")

@router.post("/index/project")
async def index_project(
    rag_system = Depends(get_rag_system)
) -> Dict[str, Any]:
    """프로젝트 인덱싱 API - 무조건 /workspace만 인덱싱, 파라미터 없음"""
    if getattr(rag_system.indexer, 'indexing_in_progress', False):
        return JSONResponse(status_code=429, content={
            "status": "indexing_in_progress",
            "message": "인덱싱 중입니다. 인덱싱이 끝난 후 다시 시도해 주세요."
        })
    try:
        logger.info(f"프로젝트 인덱싱 요청: /workspace (파라미터 없이 고정)")
        result = await rag_system.indexer.index_directory(directory_path="/workspace")
        return {
            "status": "success",
            "message": f"프로젝트 인덱싱 완료",
            "project_path": "/workspace",
            "indexing_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"프로젝트 인덱싱 실패: {e}")
        raise HTTPException(status_code=500, detail=f"프로젝트 인덱싱 실패: {str(e)}")

@router.get("/index/stats")
async def get_index_stats(
    rag_system = Depends(get_rag_system)
) -> Dict[str, Any]:
    """인덱싱 통계 조회 API"""
    
    try:
        stats = rag_system.indexer.get_indexing_stats()
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@router.post("/functions/call")
async def call_function(
    function_name: str,
    arguments: Dict[str, Any],
    function_registry = Depends(get_function_registry)
) -> Dict[str, Any]:
    """함수 직접 호출 API"""
    
    try:
        logger.info(f"함수 호출 요청: {function_name}")
        
        result = await function_registry.call_function(function_name, arguments)
        
        return {
            "status": "success",
            "function_name": function_name,
            "arguments": arguments,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"함수 호출 실패: {e}")
        raise HTTPException(status_code=500, detail=f"함수 호출 실패: {str(e)}")

@router.get("/functions")
async def list_functions(
    function_registry = Depends(get_function_registry)
) -> Dict[str, Any]:
    """사용 가능한 함수 목록 조회 API"""
    
    try:
        functions = function_registry.get_available_functions()
        
        return {
            "status": "success",
            "functions": functions,
            "total_count": len(functions)
        }
        
    except Exception as e:
        logger.error(f"함수 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"함수 목록 조회 실패: {str(e)}")

@router.post("/analyze/file")
async def analyze_file(
    file_path: str,
    include_content: bool = False
) -> Dict[str, Any]:
    """파일 분석 API"""
    
    try:
        from src.core.code_analyzer import get_code_analyzer
        analyzer = get_code_analyzer()
        
        logger.info(f"파일 분석 요청: {file_path}")
        
        analysis = await analyzer.analyze_file(file_path)
        if not analysis:
            raise HTTPException(status_code=404, detail=f"파일 분석 실패: {file_path}")
        
        result = analysis.to_dict()
        
        if include_content:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    result["file_content"] = f.read()
            except Exception as e:
                result["content_error"] = str(e)
        
        return {
            "status": "success",
            "file_path": file_path,
            "analysis": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"파일 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"파일 분석 실패: {str(e)}")

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """헬스 체크 엔드포인트"""
    
    try:
        # 각 시스템 상태 확인
        llm_manager = get_llm_manager()
        rag_system = get_rag_system()
        function_registry = get_function_registry()
        
        # 기본 상태
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": settings.VERSION,
            "components": {}
        }
        
        # LLM 상태
        model_info = llm_manager.get_model_info()
        health_status["components"]["llm"] = {
            "status": "healthy" if model_info["main_model"]["loaded"] else "degraded",
            "main_model_loaded": model_info["main_model"]["loaded"],
            "embedding_model_loaded": model_info["embedding_model"]["loaded"],
            "device": model_info["hardware"]["device"]
        }
        
        # RAG 시스템 상태
        index_stats = rag_system.indexer.get_indexing_stats()
        health_status["components"]["rag"] = {
            "status": "healthy" if index_stats.get("total_files", 0) > 0 else "warning",
            "indexed_files": index_stats.get("total_files", 0),
            "total_chunks": index_stats.get("total_chunks", 0),
            "vector_db_size": len(rag_system.vector_db.chunk_map),
            "last_index": index_stats.get("last_full_index")
        }
        
        # Function Calling 상태
        available_functions = function_registry.get_available_functions()
        health_status["components"]["functions"] = {
            "status": "healthy",
            "available_functions": len(available_functions),
            "function_names": [f["name"] for f in available_functions]
        }
        
        # 전체 상태 결정
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "unhealthy" in component_statuses:
            health_status["status"] = "unhealthy"
        elif "degraded" in component_statuses or "warning" in component_statuses:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"헬스 체크 실패: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }