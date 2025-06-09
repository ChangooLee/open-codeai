"""
Open CodeAI - 향상된 LLM 관리자
더미 모드가 완전 통합된 LLM 관리자
"""
import os
import sys
import asyncio
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from abc import ABC, abstractmethod
from transformers import GenerationConfig
import requests  # type: ignore

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModel,
        BitsAndBytesConfig
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..config import settings
from ..utils.logger import get_logger, log_performance
from ..utils.hardware import get_hardware_info, recommend_settings

logger = get_logger(__name__)

# LLMProvider 추상 클래스
class BaseLLMProvider(ABC):
    @abstractmethod
    def load(self) -> None:
        pass
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]]) -> str:
        pass
    @abstractmethod
    def device(self) -> str:
        pass

# HuggingFace Transformers Provider
class HFProvider(BaseLLMProvider):
    def __init__(self, model_path: str, model_type: str, quantize: str, device: str, max_tokens: int, temperature: float, tokenizer: Optional[Any] = None) -> None:
        self.model_path = model_path
        self.model_type = model_type
        self.quantize = quantize
        self._device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer: Optional[Any] = tokenizer
        self.model: Optional[Any] = None
        self.generation_config: Optional[Any] = None

    def load(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]]) -> str:
        raise NotImplementedError("HFProvider는 현재 사용되지 않습니다.")

    def device(self) -> str:
        return self._device

class VLLMProvider(BaseLLMProvider):
    def __init__(self, endpoint: str, api_key: Optional[str], model_id: str, max_tokens: int, temperature: float) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._device = "remote"

    def load(self) -> None:
        # No-op for remote
        pass

    def generate(self, prompt: str, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences
        logger.info(f"[VLLMProvider] 요청: endpoint={self.endpoint}/completions, model_id={self.model_id}, headers={headers}")
        try:
            resp = requests.post(f"{self.endpoint}/completions", json=payload, headers=headers, timeout=60)
            logger.info(f"[VLLMProvider] 응답 코드: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"[VLLMProvider] 응답 데이터: {data}")
            return data["choices"][0]["text"] if "choices" in data and data["choices"] else ""
        except Exception as e:
            logger.error(f"[VLLMProvider] 요청 실패: {e}")
            raise

    def device(self) -> str:
        return self._device

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int, temperature: float):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._device = "remote"
    def load(self):
        pass
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]]):
        logger.info(f"[OpenRouterProvider] generate 호출: model={self.model}, prompt={prompt[:30]}...")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature
        }
        if stop_sequences:
            payload["stop"] = stop_sequences
        try:
            logger.info(f"[OpenRouterProvider] 요청: url={self.base_url}/chat/completions, payload={payload}")
            resp = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=60)
            logger.info(f"[OpenRouterProvider] 응답 코드: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            logger.info(f"[OpenRouterProvider] 응답 데이터: {data}")
            return data["choices"][0]["message"]["content"] if "choices" in data and data["choices"] else ""
        except Exception as e:
            logger.error(f"[OpenRouterProvider] 요청 실패: {e}")
            raise
    def device(self):
        return self._device

class ModelNotLoadedException(Exception):
    """모델이 로드되지 않았을 때 발생하는 예외"""
    pass

class EnhancedLLMManager:
    """
    향상된 LLM 관리자
    - vLLMProvider(.env 기반)만 지원
    - 컨텍스트 인식 응답
    - 성능 최적화
    - 에러 복구
    """
    
    def __init__(self) -> None:
        self.llm_provider: Optional[VLLMProvider] = None
        self.openrouter_provider: Optional[OpenRouterProvider] = None
        self._device: str = "remote"
        self.mode: str = "unknown"
        self.hardware_info = get_hardware_info()
        self.recommended_settings = recommend_settings(self.hardware_info)
        self.performance_stats: Dict[str, Any] = {
            'total_requests': 0,
            'real_requests': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }
        logger.info(f"Enhanced LLM 관리자 초기화 - 디바이스: {self._device}")
        logger.info(f"권장 설정: {self.recommended_settings}")
        asyncio.create_task(self._initialize_system())

    async def _initialize_system(self) -> None:
        try:
            logger.info("Enhanced LLM 시스템 초기화 시작...")
            model_loaded = await self._load_main_model()
            if model_loaded:
                self.mode = "real"
                logger.info("실제 AI 모델 모드로 실행")
            else:
                self.mode = "basic"
                logger.warning("기본 모드로 실행 (제한된 기능)")
            logger.info(f"Enhanced LLM 시스템 초기화 완료 - 모드: {self.mode}")
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            self.mode = "error"

    async def _load_main_model(self) -> bool:
        endpoint = os.getenv("VLLM_ENDPOINT")
        api_key = os.getenv("VLLM_API_KEY")
        model_id = os.getenv("VLLM_MODEL_ID")
        use_vllm = bool(endpoint and model_id)
        logger.info(f"[LLMManager] use_vllm: {use_vllm}, endpoint: {endpoint}, model_id: {model_id}")
        vllm_failed = False
        if use_vllm:
            max_tokens = int(os.getenv("VLLM_MAX_TOKENS", 4096))
            temperature = float(os.getenv("VLLM_TEMPERATURE", 0.1))
            self.llm_provider = VLLMProvider(endpoint, api_key, model_id, max_tokens, temperature)
            try:
                test_result = self.llm_provider.generate("ping", 1, 0.0, None)
                logger.info(f"VLLMProvider 테스트 호출 성공: {test_result}")
                logger.info(f"VLLMProvider({endpoint}) 연결 및 테스트 완료: {model_id}")
                return True
            except Exception as e:
                logger.error(f"VLLMProvider 테스트 호출 실패: {e}")
                self.llm_provider = None
                vllm_failed = True
        # vllm이 실패했거나 설정이 없으면 openrouter로 즉시 fallback
        openrouter_base = os.getenv("OPENROUTER_API_BASE")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_model = os.getenv("OPENROUTER_MODEL")
        logger.info(f"[LLMManager] OpenRouterProvider fallback 진입: base={openrouter_base}, model={openrouter_model}")
        if openrouter_base and openrouter_key and openrouter_model:
            max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", 4096))
            temperature = float(os.getenv("OPENROUTER_TEMPERATURE", 0.1))
            self.openrouter_provider = OpenRouterProvider(openrouter_base, openrouter_key, openrouter_model, max_tokens, temperature)
            try:
                self.openrouter_provider.generate("ping", 1, 0.0, None)
                logger.info(f"OpenRouterProvider({openrouter_base}) 연결 및 테스트 완료: {openrouter_model}")
                return True
            except Exception as e:
                logger.error(f"OpenRouterProvider 테스트 호출 실패: {e}")
                self.openrouter_provider = None
        logger.error("VLLMProvider와 OpenRouterProvider를 위한 .env 설정이 누락되었거나 모두 실패했습니다.")
        return False

    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> str:
        logger.info(f"[LLMManager] generate_response 진입: prompt='{prompt[:50]}...', context={'있음' if context else '없음'}")
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        try:
            if context:
                enhanced_prompt = f"{context}\n\n사용자 질문: {prompt}\n\n답변:"
                logger.debug("[LLMManager] 컨텍스트 포함 프롬프트 생성")
            else:
                enhanced_prompt = prompt
                logger.debug("[LLMManager] 컨텍스트 없이 프롬프트 생성")
            if self.mode == "real" and self.llm_provider:
                try:
                    response = self.llm_provider.generate(
                        enhanced_prompt,
                        max_tokens or 256,
                        temperature if temperature is not None else 0.1,
                        stop_sequences
                    )
                    self.performance_stats['real_requests'] += 1
                    response_time = time.time() - start_time
                    self._update_performance_stats(response_time)
                    logger.info(f"[LLMManager] LLM 호출 완료, 응답 길이: {len(str(response))}")
                    return response
                except Exception:
                    pass
            if self.openrouter_provider:
                try:
                    response = self.openrouter_provider.generate(
                        enhanced_prompt,
                        max_tokens or 256,
                        temperature if temperature is not None else 0.1,
                        stop_sequences
                    )
                    response_time = time.time() - start_time
                    self._update_performance_stats(response_time)
                    logger.info(f"[LLMManager] LLM 호출 완료, 응답 길이: {len(str(response))}")
                    return response
                except Exception:
                    pass
            response = await self._generate_fallback_response(prompt)
            response_time = time.time() - start_time
            self._update_performance_stats(response_time)
            logger.info(f"[LLMManager] LLM 호출 완료, 응답 길이: {len(str(response))}")
            return response
        except Exception:
            self.performance_stats['error_count'] += 1
            logger.error("[LLMManager] LLM 호출 실패, 폴백 응답 반환")
            return await self._generate_fallback_response(prompt)

    async def _generate_fallback_response(self, prompt: str) -> str:
        await asyncio.sleep(0.5)
        return f"Open CodeAI: LLM이 로드되지 않았습니다. 관리자에게 문의하거나 vLLM 엔드포인트를 확인하세요.\n\n질문: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"

    def is_model_loaded(self) -> bool:
        return self.llm_provider is not None or self.openrouter_provider is not None

    def get_mode(self) -> str:
        return self.mode

    def get_model_info(self) -> Dict[str, Any]:
        endpoint = os.getenv("VLLM_ENDPOINT") or os.getenv("OPENROUTER_API_BASE")
        model_id = os.getenv("VLLM_MODEL_ID") or os.getenv("OPENROUTER_MODEL")
        return {
            "mode": self.mode,
            "main_model": {
                "loaded": self.is_model_loaded(),
                "endpoint": endpoint or "none",
                "model_id": model_id or "none",
                "device": self._device
            },
            "hardware": {
                "device": self._device,
                "gpu_available": False,
                "memory_gb": self.hardware_info["memory"]["total_gb"],
                "recommended_settings": self.recommended_settings
            },
            "performance": self.performance_stats
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        stats = self.performance_stats.copy()
        if stats['total_requests'] > 0:
            stats['real_mode_ratio'] = stats['real_requests'] / stats['total_requests'] 
            stats['error_rate'] = stats['error_count'] / stats['total_requests']
        else:
            stats['real_mode_ratio'] = 0.0
            stats['error_rate'] = 0.0
        return stats

    async def reload_models(self) -> None:
        logger.info("모델 재로딩 시작...")
        self.llm_provider = None
        self.openrouter_provider = None
        await self._initialize_system()
        logger.info("모델 재로딩 완료")

    async def health_check(self) -> Dict[str, Any]:
        health = {
            "status": "healthy" if self.is_model_loaded() else "unhealthy",
            "mode": self.mode,
            "timestamp": time.time(),
            "components": {
                "main_model": "healthy" if self.is_model_loaded() else "not_loaded"
            }
        }
        if self.mode == "error":
            health["status"] = "unhealthy"
        elif self.mode == "basic":
            health["status"] = "degraded"
        return health

    def _update_performance_stats(self, response_time: float) -> None:
        alpha = 0.1
        if self.performance_stats['average_response_time'] == 0:
            self.performance_stats['average_response_time'] = response_time
        else:
            self.performance_stats['average_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.performance_stats['average_response_time']
            )

    async def generate_embedding(self, text: str) -> list:
        """embedding API(9000포트)로 텍스트 임베딩 요청"""
        import aiohttp
        embedding_server_url = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:9000")
        async with aiohttp.ClientSession() as session:
            url = f"{embedding_server_url}/embedding"
            payload = {"texts": [text]}
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["embeddings"][0]

    def __del__(self) -> None:
        pass


# 전역 Enhanced LLM 관리자 인스턴스
_enhanced_llm_manager_instance = None

def get_enhanced_llm_manager() -> EnhancedLLMManager:
    """전역 Enhanced LLM 관리자 인스턴스 반환"""
    global _enhanced_llm_manager_instance
    
    if _enhanced_llm_manager_instance is None:
        _enhanced_llm_manager_instance = EnhancedLLMManager()
    
    return _enhanced_llm_manager_instance

# 하위 호환성을 위한 별칭
def get_llm_manager() -> EnhancedLLMManager:
    """하위 호환성을 위한 별칭"""
    return get_enhanced_llm_manager()