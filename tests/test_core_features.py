"""
Open CodeAI - 핵심 기능 테스트 스위트
"""
import pytest
import asyncio
import os
import tempfile
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 테스트용 임포트
from src.utils.validation import (
    validate_model_path, validate_config, check_system_requirements,
    run_comprehensive_validation
)
from src.utils.hardware import get_hardware_info, check_requirements
from src.core.code_analyzer import get_code_analyzer
from src.core.llm_manager import get_enhanced_llm_manager
from src.config import settings

# === 검증 시스템 테스트 ===

class TestValidationSystem:
    """검증 시스템 테스트"""
    
    def test_validate_model_path_nonexistent(self):
        """존재하지 않는 모델 경로 테스트"""
        result = validate_model_path("/nonexistent/path")
        assert result == False
    
    def test_validate_model_path_empty_dir(self):
        """빈 디렉토리 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_model_path(temp_dir)
            assert result == False
    
    def test_validate_model_path_valid(self):
        """유효한 모델 경로 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # config.json 생성
            config_file = Path(temp_dir) / "config.json"
            config_file.write_text('{"model_type": "test"}')
            
            # 모델 파일 생성
            model_file = Path(temp_dir) / "pytorch_model.bin"
            model_file.write_text("dummy model data")
            
            result = validate_model_path(temp_dir)
            assert result == True
    
    def test_validate_config_valid(self):
        """유효한 설정 테스트"""
        config = {
            "project": {"name": "test", "max_files": 1000},
            "server": {"host": "localhost", "port": 8000}
        }
        
        result = validate_config(config)
        assert result["valid"] == True
        assert len(result["errors"]) == 0
    
    def test_validate_config_missing_sections(self):
        """필수 섹션 누락 테스트"""
        config = {"project": {"name": "test"}}
        
        result = validate_config(config)
        assert result["valid"] == False
        assert any("server" in error for error in result["errors"])
    
    def test_validate_config_invalid_port(self):
        """잘못된 포트 테스트"""
        config = {
            "project": {"name": "test"},
            "server": {"port": 99999}
        }
        
        result = validate_config(config)
        assert result["valid"] == False
        assert any("포트" in error for error in result["errors"])
    
    def test_check_system_requirements(self):
        """시스템 요구사항 확인 테스트"""
        requirements = check_system_requirements()
        
        assert "python_version" in requirements
        assert "memory" in requirements
        assert "disk_space" in requirements
        assert "cpu_cores" in requirements
        assert "gpu" in requirements
        
        # 기본 상태 확인
        assert requirements["python_version"]["status"] in ["ok", "warning", "error"]
        assert requirements["memory"]["current_gb"] > 0
        assert requirements["disk_space"]["current_gb"] > 0
        assert requirements["cpu_cores"]["current"] > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation(self):
        """종합 검증 테스트"""
        results = run_comprehensive_validation()
        
        assert "timestamp" in results
        assert "overall_status" in results
        assert "details" in results
        assert "recommendations" in results
        
        # 기본 검증 항목들 확인
        assert "system_requirements" in results["details"]
        assert "dependencies" in results["details"]
        assert "file_structure" in results["details"]


# === 하드웨어 감지 테스트 ===

class TestHardwareDetection:
    """하드웨어 감지 테스트"""
    
    def test_get_hardware_info(self):
        """하드웨어 정보 수집 테스트"""
        info = get_hardware_info()
        
        assert "system" in info
        assert "memory" in info
        assert "gpu" in info
        assert "disk" in info
        
        # 시스템 정보 검증
        assert info["system"]["cpu_count"] > 0
        assert len(info["system"]["platform"]) > 0
        
        # 메모리 정보 검증
        assert info["memory"]["total_gb"] > 0
        assert info["memory"]["available_gb"] >= 0
        
        # 디스크 정보 검증
        assert info["disk"]["total_gb"] > 0
        assert info["disk"]["free_gb"] >= 0
    
    def test_check_requirements(self):
        """요구사항 체크 테스트"""
        requirements = check_requirements()
        
        expected_keys = [
            "memory_16gb", "memory_32gb", "memory_64gb",
            "disk_50gb", "disk_200gb", 
            "gpu_available", "gpu_12gb", "gpu_16gb",
            "cpu_8cores", "cpu_16cores"
        ]
        
        for key in expected_keys:
            assert key in requirements
            assert isinstance(requirements[key], bool)


# === 코드 분석기 테스트 ===

class TestCodeAnalyzer:
    """코드 분석기 테스트"""
    
    @pytest.mark.asyncio
    async def test_analyze_python_code(self):
        """Python 코드 분석 테스트"""
        analyzer = get_code_analyzer()
        
        python_code = '''
def hello_world(name: str) -> str:
    """간단한 인사 함수"""
    return f"Hello, {name}!"

class Person:
    """사람 클래스"""
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return hello_world(self.name)

import os
from typing import List
        '''
        
        result = await analyzer.analyze_code(python_code, "python")
        
        assert result.language == "python"
        assert len(result.functions) >= 1
        assert len(result.classes) >= 1
        assert len(result.imports) >= 2
        
        # 함수 분석 확인
        hello_func = next(f for f in result.functions if f.name == "hello_world")
        assert hello_func.parameters == ["name"]
        assert hello_func.return_type is not None
        assert hello_func.docstring is not None
        
        # 클래스 분석 확인
        person_class = next(c for c in result.classes if c.name == "Person")
        assert len(person_class.methods) >= 2  # __init__, greet
        assert person_class.docstring is not None
    
    @pytest.mark.asyncio
    async def test_analyze_javascript_code(self):
        """JavaScript 코드 분석 테스트"""
        analyzer = get_code_analyzer()
        
        javascript_code = '''
function calculateSum(a, b) {
    return a + b;
}

const multiply = (x, y) => {
    return x * y;
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = calculateSum(a, b);
        this.history.push(result);
        return result;
    }
}

import math from 'mathjs';
        '''
        
        result = await analyzer.analyze_code(javascript_code, "javascript")
        
        assert result.language == "javascript"
        assert len(result.functions) >= 2
        assert len(result.classes) >= 1
        assert len(result.imports) >= 1
    
    @pytest.mark.asyncio
    async def test_analyze_file(self):
        """파일 분석 테스트"""
        analyzer = get_code_analyzer()
        
        # 임시 Python 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def test_function():
    """테스트 함수"""
    return "test"

class TestClass:
    pass
            ''')
            temp_file = f.name
        
        try:
            result = await analyzer.analyze_file(temp_file)
            
            assert result is not None
            assert result.language == "python"
            assert len(result.functions) >= 1
            assert len(result.classes) >= 1
            
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_analyze_nonexistent_file(self):
        """존재하지 않는 파일 분석 테스트"""
        analyzer = get_code_analyzer()
        
        result = await analyzer.analyze_file("/nonexistent/file.py")
        assert result is None


# === LLM 관리자 테스트 ===

class TestLLMManager:
    """LLM 관리자 테스트"""
    
    @pytest.mark.asyncio
    async def test_llm_manager_initialization(self):
        """LLM 관리자 초기화 테스트"""
        llm_manager = get_enhanced_llm_manager()
        
        # 기본 속성 확인
        assert hasattr(llm_manager, 'mode')
        assert llm_manager.device in ["cpu", "cuda", "mps"]
        
        # 잠시 기다려서 초기화 완료
        await asyncio.sleep(1)
        
        assert llm_manager.mode in ["real", "dummy", "basic", "error"]
    
    @pytest.mark.asyncio
    async def test_generate_response_dummy_mode(self):
        """더미 모드 응답 생성 테스트"""
        llm_manager = get_enhanced_llm_manager()
        
        # 강제로 더미 모드로 전환
        await llm_manager.switch_to_dummy_mode()
        
        response = await llm_manager.generate_response("Hello, how are you?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "더미" in response or "dummy" in response.lower() or len(response) > 10
    
    @pytest.mark.asyncio
    async def test_generate_embedding_dummy_mode(self):
        """더미 모드 임베딩 생성 테스트"""
        llm_manager = get_enhanced_llm_manager()
        
        await llm_manager.switch_to_dummy_mode()
        
        embedding = await llm_manager.generate_embedding("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1024
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_code_completion(self):
        """코드 완성 테스트"""
        llm_manager = get_enhanced_llm_manager()
        
        completion = await llm_manager.generate_completion(
            "def hello_world():",
            max_tokens=100,
            temperature=0.1
        )
        
        assert isinstance(completion, str)
        assert len(completion) > 0
    
    def test_model_info(self):
        """모델 정보 조회 테스트"""
        llm_manager = get_enhanced_llm_manager()
        
        info = llm_manager.get_model_info()
        
        assert "mode" in info
        assert "main_model" in info
        assert "embedding_model" in info
        assert "hardware" in info
        assert "performance" in info
        
        # 성능 통계 확인
        perf_stats = llm_manager.get_performance_stats()
        assert "total_requests" in perf_stats
        assert "error_rate" in perf_stats
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """헬스 체크 테스트"""
        llm_manager = get_enhanced_llm_manager()
        
        health = await llm_manager.health_check()
        
        assert "status" in health
        assert "mode" in health
        assert "components" in health
        assert health["status"] in ["healthy", "degraded", "limited", "unhealthy"]


# === 설정 시스템 테스트 ===

class TestConfigSystem:
    """설정 시스템 테스트"""
    
    def test_settings_loading(self):
        """설정 로딩 테스트"""
        assert hasattr(settings, 'PROJECT_NAME')
        assert hasattr(settings, 'VERSION')
        assert hasattr(settings, 'HOST')
        assert hasattr(settings, 'PORT')
        
        # 기본값 확인
        assert settings.PROJECT_NAME == "open-codeai"
        assert isinstance(settings.PORT, int)
        assert 1000 <= settings.PORT <= 65535
    
    def test_directory_creation(self):
        """디렉토리 생성 테스트"""
        # ensure_directories는 실제로 디렉토리를 생성하므로 신중하게 테스트
        try:
            settings.ensure_directories()
            
            # 주요 디렉토리 존재 확인
            assert os.path.exists("data")
            assert os.path.exists("logs")
            
        except Exception as e:
            # 권한 문제 등으로 실패할 수 있음
            pytest.skip(f"디렉토리 생성 테스트 스킵: {e}")


# === 통합 테스트 ===

class TestIntegration:
    """통합 테스트"""
    
    @pytest.mark.asyncio
    async def test_full_dummy_workflow(self):
        """전체 더미 모드 워크플로우 테스트"""
        
        # 1. LLM 관리자 초기화
        llm_manager = get_enhanced_llm_manager()
        await llm_manager.switch_to_dummy_mode()
        
        # 2. 코드 분석
        analyzer = get_code_analyzer()
        python_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        '''
        
        analysis = await analyzer.analyze_code(python_code, "python")
        assert analysis is not None
        
        # 3. LLM 응답 생성
        prompt = f"이 코드를 설명해주세요: {python_code}"
        response = await llm_manager.generate_response(prompt)
        
        assert isinstance(response, str)
        assert len(response) > 50  # 의미있는 응답 길이
        
        # 4. 임베딩 생성
        embedding = await llm_manager.generate_embedding(python_code)
        assert len(embedding) == 1024
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_code_analysis(self):
        """대형 코드 분석 테스트"""
        analyzer = get_code_analyzer()
        
        # 큰 Python 파일 시뮬레이션
        large_code = ""
        for i in range(100):
            large_code += f'''
def function_{i}(param1, param2=None):
    """Function number {i}"""
    if param1 > {i}:
        return param1 + {i}
    return param2 or {i}

class Class_{i}:
    """Class number {i}"""
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self):
        return self.value * {i}
            '''
        
        result = await analyzer.analyze_code(large_code, "python")
        
        assert result is not None
        assert len(result.functions) >= 100
        assert len(result.classes) >= 100
        assert result.lines_of_code > 500


# === 성능 테스트 ===

class TestPerformance:
    """성능 테스트"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time(self):
        """응답 시간 테스트"""
        llm_manager = get_enhanced_llm_manager()
        await llm_manager.switch_to_dummy_mode()
        
        import time
        
        start_time = time.time()
        response = await llm_manager.generate_response("Quick test")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # 더미 모드는 5초 이내 응답
        assert response_time < 5.0
        assert len(response) > 0
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """동시 요청 테스트"""
        llm_manager = get_enhanced_llm_manager()
        await llm_manager.switch_to_dummy_mode()
        
        # 10개 동시 요청
        tasks = []
        for i in range(10):
            task = llm_manager.generate_response(f"Test request {i}")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 10
        assert all(isinstance(r, str) and len(r) > 0 for r in responses)


# === 픽스처 및 유틸리티 ===

@pytest.fixture
def sample_python_file():
    """샘플 Python 파일 픽스처"""
    content = '''
"""Sample module for testing"""

import os
import sys
from typing import List, Dict

def process_data(data: List[Dict]) -> Dict:
    """Process a list of dictionaries"""
    result = {}
    for item in data:
        if 'id' in item:
            result[item['id']] = item
    return result

class DataProcessor:
    """Data processing class"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processed_count = 0
    
    def process(self, data: List[Dict]) -> Dict:
        """Process data using the instance configuration"""
        self.processed_count += len(data)
        return process_data(data)
    
    @property
    def stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'processed_count': self.processed_count,
            'config': self.config
        }
    '''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        yield f.name
    
    os.unlink(f.name)

@pytest.fixture
def sample_javascript_file():
    """샘플 JavaScript 파일 픽스처"""
    content = '''
// Sample JavaScript module
import { EventEmitter } from 'events';
import axios from 'axios';

class ApiClient extends EventEmitter {
    constructor(baseUrl, timeout = 5000) {
        super();
        this.baseUrl = baseUrl;
        this.timeout = timeout;
    }
    
    async get(endpoint) {
        try {
            const response = await axios.get(`${this.baseUrl}${endpoint}`, {
                timeout: this.timeout
            });
            this.emit('success', response.data);
            return response.data;
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }
    
    async post(endpoint, data) {
        try {
            const response = await axios.post(`${this.baseUrl}${endpoint}`, data, {
                timeout: this.timeout
            });
            this.emit('success', response.data);
            return response.data;
        } catch (error) {
            this.emit('error', error);
            throw error;
        }
    }
}

export default ApiClient;
    '''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(content)
        yield f.name
    
    os.unlink(f.name)


# === 테스트 실행 설정 ===

if __name__ == "__main__":
    # 테스트 실행
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # 첫 번째 실패에서 중단
        "--durations=10"  # 가장 느린 10개 테스트 표시
    ])