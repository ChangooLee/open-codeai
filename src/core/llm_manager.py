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

# 더미 모드 가져오기
try:
    from .enhanced_dummy_mode import get_enhanced_dummy_llm, get_smart_dummy_rag
    DUMMY_MODE_AVAILABLE = True
except ImportError:
    DUMMY_MODE_AVAILABLE = False

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
    def load(self):
        pass
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]]) -> str:
        pass
    @abstractmethod
    def device(self) -> str:
        pass

# HuggingFace Transformers Provider
class HFProvider(BaseLLMProvider):
    def __init__(self, model_path, model_type, quantize, device, max_tokens, temperature, tokenizer=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
        self.model_path = model_path
        self.model_type = model_type
        self.quantize = quantize
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.model = None
        self.generation_config = None
        self.torch = torch
        self.GenerationConfig = GenerationConfig
        self.BitsAndBytesConfig = BitsAndBytesConfig

    def load(self):
        # 토크나이저
        if not self.tokenizer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        # 모델 로딩 옵션
        model_kwargs = {"trust_remote_code": True}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = self.torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = self.torch.float32
        # 양자화
        if self.quantize == "4bit":
            model_kwargs["quantization_config"] = self.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        elif self.quantize == "8bit":
            model_kwargs["load_in_8bit"] = True
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.model.eval()
        self.generation_config = self.GenerationConfig(
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
    def generate(self, prompt: str, max_tokens: int, temperature: float, stop_sequences: Optional[List[str]]) -> str:
        # 동기 생성
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=min(len(prompt.split())*2, self.tokenizer.model_max_length - max_tokens)).to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
        return generated_text.strip()
    def device(self) -> str:
        return self.device

class ModelNotLoadedException(Exception):
    """모델이 로드되지 않았을 때 발생하는 예외"""
    pass

class EnhancedLLMManager:
    """
    향상된 LLM 관리자
    
    - 실제 모델과 더미 모드 자동 전환
    - 컨텍스트 인식 응답
    - 성능 최적화
    - 에러 복구
    """
    
    def __init__(self):
        self.main_model = None
        self.main_tokenizer = None
        self.embedding_model = None
        self.embedding_tokenizer = None
        
        self.device = self._get_device()
        self.generation_config = None
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 더미 모드 구성 요소
        self.dummy_llm = None
        self.dummy_rag = None
        self.mode = "unknown"  # "real", "dummy", "hybrid"
        
        # 하드웨어 정보 및 권장 설정
        self.hardware_info = get_hardware_info()
        self.recommended_settings = recommend_settings(self.hardware_info)
        
        # 성능 메트릭
        self.performance_stats = {
            'total_requests': 0,
            'dummy_requests': 0,
            'real_requests': 0,
            'average_response_time': 0.0,
            'error_count': 0
        }
        
        logger.info(f"Enhanced LLM 관리자 초기화 - 디바이스: {self.device}")
        logger.info(f"권장 설정: {self.recommended_settings}")
        
        # 비동기 초기화
        asyncio.create_task(self._initialize_system())
        
        self.llm_provider = None
    
    def _get_device(self) -> str:
        """최적 디바이스 결정"""
        if not HAS_TORCH:
            return "cpu"
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    async def _initialize_system(self):
        """전체 시스템 초기화"""
        try:
            logger.info("Enhanced LLM 시스템 초기화 시작...")
            
            # 1. 더미 모드 초기화 (항상 사용 가능하도록)
            await self._initialize_dummy_mode()
            
            # 2. 실제 모델 로딩 시도
            model_loaded = await self._initialize_real_models()
            
            # 3. 모드 결정
            if model_loaded:
                self.mode = "real"
                logger.success("실제 AI 모델 모드로 실행")
            elif self.dummy_llm:
                self.mode = "dummy"
                logger.warning("더미 모드로 실행 (모델 미로드)")
            else:
                self.mode = "basic"
                logger.warning("기본 모드로 실행 (제한된 기능)")
            
            logger.success(f"Enhanced LLM 시스템 초기화 완료 - 모드: {self.mode}")
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            self.mode = "error"
    
    async def _initialize_dummy_mode(self):
        """더미 모드 초기화"""
        try:
            if DUMMY_MODE_AVAILABLE:
                self.dummy_llm = get_enhanced_dummy_llm()
                self.dummy_rag = get_smart_dummy_rag()
                logger.success("더미 모드 구성 요소 초기화 완료")
            else:
                logger.warning("더미 모드 모듈을 찾을 수 없습니다")
        except Exception as e:
            logger.error(f"더미 모드 초기화 실패: {e}")
    
    async def _initialize_real_models(self) -> bool:
        """실제 모델들 초기화"""
        try:
            if not HAS_TORCH:
                logger.warning("PyTorch가 설치되지 않았습니다")
                return False
            
            # 메인 모델 로딩
            main_loaded = False
            if settings.llm and settings.llm.main_model:
                main_loaded = await self._load_main_model()
            
            # 임베딩 모델 로딩
            embedding_loaded = False
            if settings.llm and settings.llm.embedding_model:
                embedding_loaded = await self._load_embedding_model()
            
            return main_loaded or embedding_loaded
            
        except Exception as e:
            logger.error(f"실제 모델 초기화 실패: {e}")
            return False
    
    async def _load_main_model(self) -> bool:
        model_path = settings.llm.main_model.path
        model_type = getattr(settings.llm.main_model, 'type', 'qwen2.5-coder')
        quantize = getattr(settings.llm.main_model, 'quantize', 'none')
        device = getattr(settings.llm.main_model, 'device', 'auto')
        max_tokens = getattr(settings.llm.main_model, 'max_tokens', 4096)
        temperature = getattr(settings.llm.main_model, 'temperature', 0.1)
        # provider는 항상 HFProvider만 사용
        self.llm_provider = HFProvider(model_path, model_type, quantize, device, max_tokens, temperature)
        try:
            self.llm_provider.load()
            self.main_model = self.llm_provider.model
            self.main_tokenizer = getattr(self.llm_provider, 'tokenizer', None)
            logger.success(f"LLMProvider({type(self.llm_provider).__name__}) 로딩 완료: {model_path}")
            return True
        except Exception as e:
            logger.error(f"LLMProvider 로딩 실패: {e}")
            self.main_model = None
            self.main_tokenizer = None
            return False
    
    @log_performance("embedding_model_loading")
    async def _load_embedding_model(self) -> bool:
        """임베딩 모델 로딩"""
        model_path = settings.llm.embedding_model.path
        
        if not os.path.exists(model_path):
            logger.warning(f"임베딩 모델 경로를 찾을 수 없습니다: {model_path}")
            return False
        
        try:
            logger.info(f"임베딩 모델 로딩 중: {model_path}")
            
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.embedding_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device != "cuda":
                self.embedding_model = self.embedding_model.to(self.device)
            
            self.embedding_model.eval()
            
            logger.success(f"임베딩 모델 로딩 완료: {settings.llm.embedding_model.name}")
            return True
            
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {e}")
            self.embedding_model = None
            self.embedding_tokenizer = None
            return False
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> str:
        """
        지능형 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            stop_sequences: 중지 시퀀스
            context: 추가 컨텍스트
            
        Returns:
            생성된 응답
        """
        
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        try:
            # 컨텍스트 통합
            if context:
                enhanced_prompt = f"{context}\n\n사용자 질문: {prompt}\n\n답변:"
            else:
                enhanced_prompt = prompt
            
            # 모드별 처리
            if self.mode == "real" and self.main_model:
                response = await self._generate_real_response(
                    enhanced_prompt, max_tokens, temperature, stop_sequences
                )
                self.performance_stats['real_requests'] += 1
                
            elif self.mode == "dummy" and self.dummy_llm:
                response = await self.dummy_llm.generate_smart_response(enhanced_prompt)
                self.performance_stats['dummy_requests'] += 1
                
            else:
                response = await self._generate_fallback_response(prompt)
                self.performance_stats['dummy_requests'] += 1
            
            # 성능 통계 업데이트
            response_time = time.time() - start_time
            self._update_performance_stats(response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            self.performance_stats['error_count'] += 1
            
            # 폴백 처리
            if self.dummy_llm:
                return await self.dummy_llm.generate_smart_response(prompt)
            else:
                return await self._generate_fallback_response(prompt)
    
    async def _generate_real_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int],
        temperature: Optional[float],
        stop_sequences: Optional[List[str]]
    ) -> str:
        """실제 모델로 응답 생성"""
        
        try:
            # 생성 설정 준비
            generation_config = GenerationConfig.from_dict(self.generation_config.to_dict())
            
            if max_tokens:
                generation_config.max_new_tokens = min(max_tokens, 4096)
            if temperature is not None:
                generation_config.temperature = temperature
            
            # 비동기 생성 실행
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor, 
                self._generate_text_sync, 
                prompt, 
                generation_config,
                stop_sequences
            )
            
            return response
            
        except Exception as e:
            logger.error(f"실제 모델 생성 실패: {e}")
            raise
    
    def _generate_text_sync(
        self, 
        prompt: str, 
        generation_config: GenerationConfig,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """동기 텍스트 생성"""
        
        # provider가 있으면 provider로 위임
        if self.llm_provider:
            return self.llm_provider.generate(prompt, generation_config.max_new_tokens, generation_config.temperature, stop_sequences)
        
        with self.model_lock:
            try:
                # 토크나이징
                inputs = self.main_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(
                        len(prompt.split()) * 2, 
                        self.main_tokenizer.model_max_length - generation_config.max_new_tokens
                    )
                ).to(self.device)
                
                # 생성
                with torch.no_grad():
                    outputs = self.main_model.generate(
                        **inputs,
                        generation_config=generation_config,
                        pad_token_id=self.main_tokenizer.pad_token_id,
                        eos_token_id=self.main_tokenizer.eos_token_id,
                        do_sample=True
                    )
                
                # 디코딩
                generated_text = self.main_tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # 중지 시퀀스 처리
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                            break
                
                return generated_text.strip()
                
            except Exception as e:
                logger.error(f"동기 생성 실패: {e}")
                raise
    
    async def generate_completion(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """코드 완성 생성"""
        
        # FIM (Fill-in-the-Middle) 처리
        if "<PRE>" in prompt and "<SUF>" in prompt and "<MID>" in prompt:
            parts = prompt.split("<PRE>")[1].split("<SUF>")
            prefix = parts[0]
            suffix_and_mid = parts[1].split("<MID>")
            suffix = suffix_and_mid[0] 
            
            completion_prompt = f"# Complete the following code:\n{prefix}# TODO: Complete here\n{suffix}"
        else:
            completion_prompt = prompt
        
        # 코드 완성 최적화 설정
        if temperature is None:
            temperature = 0.1
        if max_tokens is None:
            max_tokens = 256
        
        code_stop_sequences = ["\n\n", "```", "def ", "class ", "import ", "from "]
        if stop_sequences:
            code_stop_sequences.extend(stop_sequences)
        
        return await self.generate_response(
            completion_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=code_stop_sequences
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """임베딩 생성"""
        
        try:
            if self.mode == "real" and self.embedding_model:
                return await self._generate_real_embedding(text)
            elif self.dummy_llm:
                return await self.dummy_llm.generate_smart_embedding(text)
            else:
                return await self._generate_fallback_embedding(text)
                
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            # 폴백
            if self.dummy_llm:
                return await self.dummy_llm.generate_smart_embedding(text)
            else:
                return await self._generate_fallback_embedding(text)
    
    async def _generate_real_embedding(self, text: str) -> List[float]:
        """실제 모델로 임베딩 생성"""
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            self._generate_embedding_sync,
            text
        )
        
        return embedding
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """동기 임베딩 생성"""
        
        with self.model_lock:
            try:
                inputs = self.embedding_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # Mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    mean_embeddings = sum_embeddings / sum_mask
                    
                    # 정규화
                    embeddings_normalized = F.normalize(mean_embeddings, p=2, dim=1)
                    
                    return embeddings_normalized[0].cpu().numpy().tolist()
                    
            except Exception as e:
                logger.error(f"동기 임베딩 생성 실패: {e}")
                raise
    
    async def _generate_fallback_response(self, prompt: str) -> str:
        """기본 폴백 응답"""
        
        await asyncio.sleep(0.5)  # 실제 AI처럼 지연
        
        return f"""안녕하세요! Open CodeAI가 기본 모드로 응답드립니다.

현재 질문: "{prompt[:100]}{'...' if len(prompt) > 100 else ''}"

**현재 상태:**
- 실제 AI 모델: 미로드
- 더미 모드: 제한적 사용 가능

**모든 기능을 사용하려면:**
1. 모델 다운로드: `python scripts/download_models.py`
2. 서버 재시작: `./start.sh restart`

기본적인 도움은 계속 제공할 수 있습니다. 구체적인 질문을 해주세요!"""
    
    async def _generate_fallback_embedding(self, text: str) -> List[float]:
        """기본 폴백 임베딩"""
        
        await asyncio.sleep(0.1)
        
        # 간단한 해시 기반 임베딩
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        embedding = []
        for i in range(0, len(hash_hex), 2):
            val = int(hash_hex[i:i+2], 16) / 255.0 * 2 - 1
            embedding.append(val)
        
        while len(embedding) < 1024:
            embedding.extend(embedding[:min(16, 1024 - len(embedding))])
        
        return embedding[:1024]
    
    def _update_performance_stats(self, response_time: float):
        """성능 통계 업데이트"""
        
        # 이동 평균으로 응답 시간 계산
        alpha = 0.1  # 평활화 계수
        if self.performance_stats['average_response_time'] == 0:
            self.performance_stats['average_response_time'] = response_time
        else:
            self.performance_stats['average_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.performance_stats['average_response_time']
            )
    
    # 유틸리티 메서드들
    
    def is_model_loaded(self) -> bool:
        """실제 모델 로딩 상태 확인"""
        return self.main_model is not None
    
    def is_embedding_model_loaded(self) -> bool:
        """임베딩 모델 로딩 상태 확인"""
        return self.embedding_model is not None
    
    def get_mode(self) -> str:
        """현재 모드 반환"""
        return self.mode
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "mode": self.mode,
            "main_model": {
                "loaded": self.is_model_loaded(),
                "name": settings.llm.main_model.name if (settings.llm and hasattr(settings.llm, 'main_model')) else "none",
                "path": settings.llm.main_model.path if (settings.llm and hasattr(settings.llm, 'main_model')) else "none",
                "device": self.device
            },
            "embedding_model": {
                "loaded": self.is_embedding_model_loaded(),
                "name": settings.llm.embedding_model.name if (settings.llm and hasattr(settings.llm, 'embedding_model')) else "none",
                "path": settings.llm.embedding_model.path if (settings.llm and hasattr(settings.llm, 'embedding_model')) else "none",
                "device": self.device
            },
            "hardware": {
                "device": self.device,
                "gpu_available": self.device == "cuda",
                "memory_gb": self.hardware_info["memory"]["total_gb"],
                "recommended_settings": self.recommended_settings
            },
            "performance": self.performance_stats,
            "dummy_mode_available": DUMMY_MODE_AVAILABLE and self.dummy_llm is not None
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        
        # 추가 계산된 메트릭
        if stats['total_requests'] > 0:
            stats['dummy_mode_ratio'] = stats['dummy_requests'] / stats['total_requests']
            stats['real_mode_ratio'] = stats['real_requests'] / stats['total_requests'] 
            stats['error_rate'] = stats['error_count'] / stats['total_requests']
        else:
            stats['dummy_mode_ratio'] = 0.0
            stats['real_mode_ratio'] = 0.0
            stats['error_rate'] = 0.0
        
        return stats
    
    async def reload_models(self):
        """모델 재로딩"""
        logger.info("모델 재로딩 시작...")
        
        # 기존 모델 해제
        if self.main_model:
            del self.main_model
            self.main_model = None
        
        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None
        
        # GPU 메모리 정리
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 재초기화
        await self._initialize_system()
        
        logger.success("모델 재로딩 완료")
    
    async def switch_to_dummy_mode(self):
        """더미 모드로 강제 전환"""
        logger.info("더미 모드로 전환 중...")
        
        # 실제 모델 언로드
        if self.main_model:
            del self.main_model
            self.main_model = None
        
        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None
        
        # GPU 메모리 정리
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 더미 모드 활성화
        await self._initialize_dummy_mode()
        self.mode = "dummy"
        
        logger.success("더미 모드로 전환 완료")
    
    async def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        
        health = {
            "status": "healthy",
            "mode": self.mode,
            "timestamp": time.time(),
            "components": {}
        }
        
        # 메인 모델 상태
        if self.is_model_loaded():
            health["components"]["main_model"] = "healthy"
        elif self.mode == "dummy":
            health["components"]["main_model"] = "dummy_mode"
        else:
            health["components"]["main_model"] = "not_loaded"
        
        # 임베딩 모델 상태
        if self.is_embedding_model_loaded():
            health["components"]["embedding_model"] = "healthy"
        elif self.dummy_llm:
            health["components"]["embedding_model"] = "dummy_mode"
        else:
            health["components"]["embedding_model"] = "not_loaded"
        
        # 더미 모드 상태
        if self.dummy_llm:
            health["components"]["dummy_mode"] = "available"
        else:
            health["components"]["dummy_mode"] = "not_available"
        
        # 전체 상태 결정
        if self.mode == "error":
            health["status"] = "unhealthy"
        elif self.mode == "basic":
            health["status"] = "degraded"
        elif self.mode == "dummy":
            health["status"] = "limited"
        
        return health
    
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


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