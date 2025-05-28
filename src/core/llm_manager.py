"""
Open CodeAI - LLM 관리자 구현
로컬 모델 로딩, 추론, 임베딩 생성을 담당
"""
import os
import asyncio
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModel,
        GenerationConfig, BitsAndBytesConfig
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..config import settings
from ..utils.logger import get_logger, log_performance
from ..utils.hardware import get_hardware_info, recommend_settings

logger = get_logger(__name__)

class ModelNotLoadedException(Exception):
    """모델이 로드되지 않았을 때 발생하는 예외"""
    pass

class LLMManager:
    """
    로컬 LLM 관리자
    
    - 모델 로딩 및 관리
    - 텍스트 생성 및 완성
    - 임베딩 생성
    - GPU/CPU 최적화
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
        
        # 하드웨어 정보 및 권장 설정
        self.hardware_info = get_hardware_info()
        self.recommended_settings = recommend_settings(self.hardware_info)
        
        logger.info(f"LLM 관리자 초기화 - 디바이스: {self.device}")
        logger.info(f"권장 설정: {self.recommended_settings}")
        
        # 모델 로딩 (비동기)
        asyncio.create_task(self._initialize_models())
    
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
    
    async def _initialize_models(self):
        """모델들을 비동기로 초기화"""
        try:
            logger.info("모델 초기화 시작...")
            
            # 메인 모델 로딩
            if settings.llm and settings.llm.main_model:
                await self._load_main_model()
            else:
                logger.warning("메인 모델 설정이 없습니다. 더미 모드로 실행됩니다.")
            
            # 임베딩 모델 로딩  
            if settings.llm and settings.llm.embedding_model:
                await self._load_embedding_model()
            else:
                logger.warning("임베딩 모델 설정이 없습니다.")
                
            logger.success("모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            # 실패해도 서버는 계속 실행 (더미 모드)
    
    @log_performance("main_model_loading")
    async def _load_main_model(self):
        """메인 LLM 모델 로딩"""
        if not HAS_TORCH:
            logger.warning("PyTorch가 설치되지 않았습니다. 더미 모드로 실행됩니다.")
            return
        
        model_path = settings.llm.main_model.path
        
        if not os.path.exists(model_path):
            logger.error(f"모델 경로를 찾을 수 없습니다: {model_path}")
            return
        
        try:
            logger.info(f"메인 모델 로딩 중: {model_path}")
            
            # 토크나이저 로딩
            self.main_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # 패딩 토큰 설정
            if self.main_tokenizer.pad_token is None:
                self.main_tokenizer.pad_token = self.main_tokenizer.eos_token
            
            # 모델 로딩 옵션 설정
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            # GPU 메모리 최적화
            if self.device == "cuda" and self.recommended_settings["gpu"]["memory_fraction"] < 1.0:
                model_kwargs["max_memory"] = {
                    0: f"{int(self.recommended_settings['memory_limit_gb'] * 0.8)}GB"
                }
            
            # 양자화 설정 (메모리 절약)
            if self.device == "cuda" and self.hardware_info["gpu"]["devices"]:
                gpu_memory = self.hardware_info["gpu"]["devices"][0]["memory_gb"]
                if gpu_memory < 24:  # 24GB 미만이면 양자화 사용
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
            
            # 모델 로딩
            self.main_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # 평가 모드로 설정
            self.main_model.eval()
            
            # 생성 설정
            self.generation_config = GenerationConfig(
                max_new_tokens=settings.llm.main_model.max_tokens,
                temperature=settings.llm.main_model.temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.main_tokenizer.pad_token_id,
                eos_token_id=self.main_tokenizer.eos_token_id,
            )
            
            logger.success(f"메인 모델 로딩 완료: {settings.llm.main_model.name}")
            
        except Exception as e:
            logger.error(f"메인 모델 로딩 실패: {e}")
            self.main_model = None
            self.main_tokenizer = None
    
    @log_performance("embedding_model_loading")
    async def _load_embedding_model(self):
        """임베딩 모델 로딩"""
        if not HAS_TORCH:
            return
        
        model_path = settings.llm.embedding_model.path
        
        if not os.path.exists(model_path):
            logger.error(f"임베딩 모델 경로를 찾을 수 없습니다: {model_path}")
            return
        
        try:
            logger.info(f"임베딩 모델 로딩 중: {model_path}")
            
            # 토크나이저 로딩
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # 모델 로딩
            self.embedding_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # 모델을 디바이스로 이동
            if self.device != "cuda":  # device_map이 auto가 아닐 때만
                self.embedding_model = self.embedding_model.to(self.device)
            
            self.embedding_model.eval()
            
            logger.success(f"임베딩 모델 로딩 완료: {settings.llm.embedding_model.name}")
            
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 실패: {e}")
            self.embedding_model = None
            self.embedding_tokenizer = None
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        텍스트 생성 (채팅 응답용)
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            stop_sequences: 중지 시퀀스
            
        Returns:
            생성된 텍스트
        """
        
        # 더미 모드 (모델이 로딩되지 않은 경우)
        if not self.main_model or not HAS_TORCH:
            return await self._generate_dummy_response(prompt)
        
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
            logger.error(f"텍스트 생성 실패: {e}")
            return await self._generate_dummy_response(prompt)
    
    def _generate_text_sync(
        self, 
        prompt: str, 
        generation_config: GenerationConfig,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """동기 텍스트 생성 (스레드풀에서 실행)"""
        
        with self.model_lock:
            try:
                # 토크나이징
                inputs = self.main_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(len(prompt.split()) * 2, self.main_tokenizer.model_max_length - generation_config.max_new_tokens)
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
        """
        코드 완성 생성 (자동완성용)
        
        Args:
            prompt: 완성할 프롬프트 (FIM 포맷 포함 가능)
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            stop_sequences: 중지 시퀀스
            
        Returns:
            완성된 텍스트
        """
        
        # FIM (Fill-in-the-Middle) 처리
        if "<PRE>" in prompt and "<SUF>" in prompt and "<MID>" in prompt:
            # FIM 포맷 파싱
            parts = prompt.split("<PRE>")[1].split("<SUF>")
            prefix = parts[0]
            suffix_and_mid = parts[1].split("<MID>")
            suffix = suffix_and_mid[0] 
            
            # 코드 완성에 최적화된 프롬프트 생성
            completion_prompt = f"# Complete the following code:\n{prefix}# TODO: Complete here\n{suffix}"
        else:
            completion_prompt = prompt
        
        # 코드 완성에 적합한 설정 조정
        if temperature is None:
            temperature = 0.1  # 코드 완성은 낮은 온도 사용
        
        if max_tokens is None:
            max_tokens = 256  # 코드 완성은 짧게
        
        # 코드 완성용 중지 시퀀스 추가
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
        """
        텍스트 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            임베딩 벡터
        """
        
        # 더미 모드
        if not self.embedding_model or not HAS_TORCH:
            return await self._generate_dummy_embedding(text)
        
        try:
            # 비동기 임베딩 생성
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self._generate_embedding_sync,
                text
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return await self._generate_dummy_embedding(text)
    
    def _generate_embedding_sync(self, text: str) -> List[float]:
        """동기 임베딩 생성"""
        
        with self.model_lock:
            try:
                # 토크나이징
                inputs = self.embedding_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    
                    # Mean pooling
                    embeddings = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']
                    
                    # 마스크된 평균 계산
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
    
    # 더미 모드 함수들 (모델이 없을 때 사용)
    
    async def _generate_dummy_response(self, prompt: str) -> str:
        """더미 응답 생성 (개발/테스트용)"""
        
        # 실제 AI처럼 보이는 지연 시간
        await asyncio.sleep(0.5)
        
        # 프롬프트 분석해서 적절한 더미 응답 생성
        if "code" in prompt.lower() or "function" in prompt.lower():
            return """```python
def example_function():
    \"\"\"이것은 더미 코드 생성 예시입니다.\"\"\"
    return "Hello from Open CodeAI!"
```

이 코드는 더미 모드에서 생성된 예시입니다. 실제 모델을 로드하려면 모델 파일을 설정하세요."""
        
        elif "explain" in prompt.lower():
            return "이것은 Open CodeAI의 더미 모드 응답입니다. 실제 AI 모델이 로드되면 더 정확한 설명을 제공할 수 있습니다."
        
        else:
            return "안녕하세요! Open CodeAI가 더미 모드로 실행 중입니다. 모델을 로드하면 더 나은 응답을 제공할 수 있습니다."
    
    async def _generate_dummy_embedding(self, text: str) -> List[float]:
        """더미 임베딩 생성"""
        
        await asyncio.sleep(0.1)
        
        # 텍스트 기반의 의사 랜덤 임베딩 생성
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # 1024차원 임베딩 생성
        embedding = []
        for i in range(0, len(hash_hex), 2):
            # 16진수를 -1~1 범위의 float로 변환
            val = int(hash_hex[i:i+2], 16) / 255.0 * 2 - 1
            embedding.append(val)
        
        # 1024차원까지 확장
        while len(embedding) < 1024:
            embedding.extend(embedding[:min(16, 1024 - len(embedding))])
        
        return embedding[:1024]
    
    # 유틸리티 메서드들
    
    def is_model_loaded(self) -> bool:
        """모델 로딩 상태 확인"""
        return self.main_model is not None
    
    def is_embedding_model_loaded(self) -> bool:
        """임베딩 모델 로딩 상태 확인"""
        return self.embedding_model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "main_model": {
                "loaded": self.is_model_loaded(),
                "name": settings.llm.main_model.name if settings.llm else "none",
                "path": settings.llm.main_model.path if settings.llm else "none",
                "device": self.device
            },
            "embedding_model": {
                "loaded": self.is_embedding_model_loaded(),
                "name": settings.llm.embedding_model.name if settings.llm else "none", 
                "path": settings.llm.embedding_model.path if settings.llm else "none",
                "device": self.device
            },
            "hardware": {
                "device": self.device,
                "gpu_available": self.device == "cuda",
                "memory_gb": self.hardware_info["memory"]["total_gb"],
                "recommended_settings": self.recommended_settings
            }
        }
    
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
        
        # 모델 재로딩
        await self._initialize_models()
        
        logger.success("모델 재로딩 완료")
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()


# 전역 LLM 관리자 인스턴스 (싱글톤 패턴)
_llm_manager_instance = None

def get_llm_manager() -> LLMManager:
    """전역 LLM 관리자 인스턴스 반환"""
    global _llm_manager_instance
    
    if _llm_manager_instance is None:
        _llm_manager_instance = LLMManager()
    
    return _llm_manager_instance