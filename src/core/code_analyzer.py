"""
Open CodeAI - 코드 분석기
Tree-sitter + Python AST 기반의 강력한 코드 분석 시스템
"""
import ast
import re
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Match, Sequence, cast, TypeAlias, Awaitable, Pattern, TypedDict, TypeVar, Protocol
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from src.utils.mapper_xml_parser import parse_mapper_xml
from src.utils.logger import get_logger
from .tree_sitter_analyzer import TreeSitterAnalyzer, Language
import subprocess

logger = get_logger(__name__)

# Type aliases
FunctionNode: TypeAlias = Union[ast.FunctionDef, ast.AsyncFunctionDef]
AnalysisResult: TypeAlias = Dict[str, Any]

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    result: AnalysisResult
    timestamp: datetime
    hash: str

@dataclass
class CodeMetrics:
    """코드 품질 메트릭"""
    complexity: int = 0
    maintainability_index: float = 0.0
    cognitive_complexity: int = 0
    cyclomatic_complexity: int = 0
    loc: int = 0
    comment_ratio: float = 0.0
    duplication_ratio: float = 0.0
    test_coverage: float = 0.0
    code_smells: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    code_quality_score: float = 0.0
    technical_debt: float = 0.0
    code_churn: float = 0.0
    test_quality: float = 0.0

@dataclass
class Dependency:
    """의존성 정보"""
    source: str
    target: str
    type: str
    line: int
    weight: float = 1.0
    bidirectional: bool = False

@dataclass
class CodeElement:
    """코드 요소 기본 클래스"""
    name: str
    start_line: int
    end_line: int
    element_type: str
    content: str
    language: str
    dependencies: List[Dependency] = field(default_factory=list)
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CodeAnalyzer:
    """코드 분석기"""
    
    def __init__(self) -> None:
        # CPU 코어 수에 기반한 최적의 워커 수 설정
        cpu_count = os.cpu_count() or 4
        self.thread_executor = ThreadPoolExecutor(max_workers=cpu_count * 2)
        self.process_executor = ProcessPoolExecutor(max_workers=cpu_count)
        
        # Tree-sitter 초기화
        self._initialize_tree_sitter()
        
        self.tree_sitter_analyzer = TreeSitterAnalyzer()
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl = timedelta(hours=1)
        self.max_cache_size = 1000
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.analysis_queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue(maxsize=cpu_count * 4)
        self.result_queue: asyncio.Queue[AnalysisResult] = asyncio.Queue(maxsize=cpu_count * 4)
        self.worker_tasks: List[asyncio.Task[None]] = []
        self.stop_event = asyncio.Event()
        self._cache_lock = asyncio.Lock()
        self._file_locks: Dict[str, asyncio.Lock] = {}
        self._analysis_semaphore = asyncio.Semaphore(cpu_count * 2)
        self._retry_count = 3
        self._retry_delay = 1.0
        self._chunk_size = 1024 * 1024  # 1MB
        self._max_file_size = 10 * 1024 * 1024  # 10MB
        self._supported_extensions = {'.py', '.java', '.xml', '.js', '.ts', '.cpp', '.c', '.h', '.hpp'}
        self._default_result: AnalysisResult = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'calls': [],
            'language': '',
            'dependencies': [],
            'metrics': asdict(CodeMetrics())
        }
        self._analysis_priority = {
            '.py': ['tree_sitter', 'python_ast'],
            '.java': ['tree_sitter'],
            '.xml': ['mybatis'],
            '.js': ['tree_sitter'],
            '.ts': ['tree_sitter'],
            '.cpp': ['tree_sitter'],
            '.c': ['tree_sitter'],
            '.h': ['tree_sitter'],
            '.hpp': ['tree_sitter']
        }
        self._file_type_languages = {
            '.py': 'python',
            '.java': 'java',
            '.xml': 'xml',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp'
        }
        
    def _initialize_tree_sitter(self) -> None:
        """Tree-sitter 초기화"""
        try:
            from tree_sitter_languages import get_parser
            
            # 언어 정의
            languages = {
                'python': 'python',
                'java': 'java',
                'javascript': 'javascript',
                'typescript': 'typescript',
                'cpp': 'cpp',
                'c': 'c'
            }
            
            # 각 언어별 파서 초기화
            self.parsers = {}
            for lang_name in languages.values():
                try:
                    parser = get_parser(lang_name)
                    if parser:
                        self.parsers[lang_name] = parser
                        logger.info(f"Tree-sitter {lang_name} 파서 초기화 완료")
                    else:
                        logger.warning(f"Tree-sitter {lang_name} 파서를 찾을 수 없습니다")
                except Exception as e:
                    logger.error(f"Tree-sitter {lang_name} 파서 초기화 실패: {e}")
                    
        except Exception as e:
            logger.error(f"Tree-sitter 초기화 실패: {e}")
            
    def _get_file_language(self, file_path: str) -> str:
        """파일 타입에 따른 언어 반환"""
        ext = os.path.splitext(file_path)[1].lower()
        return self._file_type_languages.get(ext, '')
        
    def _create_error_result(self, error_msg: str) -> AnalysisResult:
        """에러 결과 생성"""
        result = self._default_result.copy()
        result['error'] = error_msg
        return result
        
    def _merge_analysis_results(self, results: List[AnalysisResult]) -> AnalysisResult:
        """분석 결과 병합"""
        merged = self._default_result.copy()
        
        for result in results:
            if not self._validate_analysis_result(result):
                continue
                
            # 리스트 필드 병합
            for field in ['functions', 'classes', 'imports', 'variables', 'calls', 'dependencies']:
                if field in result:
                    merged[field].extend(result[field])
                    
            # 메트릭스 병합
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if key in merged['metrics']:
                        merged['metrics'][key] += value
                    else:
                        merged['metrics'][key] = value
                        
            # 언어 설정
            if result.get('language'):
                merged['language'] = result['language']
                
        return merged
        
    def _is_supported_file(self, file_path: str) -> bool:
        """지원되는 파일인지 확인"""
        return os.path.splitext(file_path)[1].lower() in self._supported_extensions
        
    def _validate_analysis_result(self, result: AnalysisResult) -> bool:
        """분석 결과 유효성 검증"""
        if not isinstance(result, dict):
            return False
            
        # 필수 필드 확인
        required_fields = {'functions', 'classes', 'imports', 'variables', 'calls', 'language', 'dependencies', 'metrics'}
        if not all(field in result for field in required_fields):
            return False
            
        # 필드 타입 확인
        if not isinstance(result['functions'], list) or \
           not isinstance(result['classes'], list) or \
           not isinstance(result['imports'], list) or \
           not isinstance(result['variables'], list) or \
           not isinstance(result['calls'], list) or \
           not isinstance(result['language'], str) or \
           not isinstance(result['dependencies'], list) or \
           not isinstance(result['metrics'], dict):
            return False
            
        return True
        
    def _get_file_lock(self, file_path: str) -> asyncio.Lock:
        """파일별 락 가져오기"""
        if file_path not in self._file_locks:
            self._file_locks[file_path] = asyncio.Lock()
        return self._file_locks[file_path]
        
    async def _get_cached_result(self, file_path: str) -> Optional[AnalysisResult]:
        """캐시된 결과 조회 (스레드 안전)"""
        async with self._cache_lock:
            file_hash = self._generate_file_hash(file_path)
            if not file_hash:
                return None
                
            cache_entry = self.cache.get(file_path)
            if cache_entry and cache_entry.hash == file_hash:
                if datetime.now() - cache_entry.timestamp < self.cache_ttl:
                    return cache_entry.result
                else:
                    self.cache.pop(file_path, None)
            return None
            
    async def _cache_result(self, file_path: str, result: AnalysisResult) -> None:
        """결과 캐싱 (스레드 안전)"""
        async with self._cache_lock:
            file_hash = self._generate_file_hash(file_path)
            if not file_hash:
                return
                
            # 캐시 크기 제한
            if len(self.cache) >= self.max_cache_size:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                self.cache.pop(oldest_key, None)
                
            self.cache[file_path] = CacheEntry(
                result=result,
                timestamp=datetime.now(),
                hash=file_hash
            )
            
    async def _read_file_chunked(self, file_path: str) -> str:
        """청크 단위로 파일 읽기"""
        content = []
        total_size = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(self._chunk_size)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > self._max_file_size:
                        raise ValueError(f'File too large: {file_path}')
                    content.append(chunk)
            return ''.join(content)
        except UnicodeDecodeError:
            # UTF-8 디코딩 실패 시 다른 인코딩 시도
            encodings = ['latin-1', 'cp949', 'euc-kr']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f'Unsupported file encoding: {file_path}')
            
    def _get_analysis_priority(self, file_path: str) -> List[str]:
        """파일 타입별 분석 우선순위 반환"""
        ext = os.path.splitext(file_path)[1].lower()
        return self._analysis_priority.get(ext, ['tree_sitter'])
        
    async def _analyze_with_method(self, file_path: str, method: str, language: Optional[str] = None) -> Optional[AnalysisResult]:
        """지정된 방법으로 파일 분석"""
        try:
            if method == 'tree_sitter' and language:
                return await self.tree_sitter_analyzer.analyze(file_path, language)
            elif method == 'python_ast' and file_path.endswith('.py'):
                content = await self._read_file_chunked(file_path)
                return self._analyze_python_file(content)
            elif method == 'mybatis' and file_path.endswith('.xml'):
                content = await self._read_file_chunked(file_path)
                queries = parse_mapper_xml(content)
                return {
                    'functions': [],
                    'classes': [],
                    'imports': [],
                    'variables': [],
                    'calls': [],
                    'language': 'xml',
                    'dependencies': [],
                    'metrics': asdict(CodeMetrics()),
                    'queries': queries
                }
        except Exception as e:
            logger.warning(f"{method} 분석 실패({file_path}): {e}")
        return None
        
    async def analyze_file(self, file_path: str, language: Optional[str] = None) -> AnalysisResult:
        """파일 분석"""
        # 파일별 락을 사용하여 동시 분석 방지
        async with self._get_file_lock(file_path):
            # 캐시된 결과 확인
            cached_result = await self._get_cached_result(file_path)
            if cached_result:
                return cached_result
                
            # 파일 존재 및 접근 권한 확인
            if not os.path.exists(file_path):
                return self._create_error_result(f'File not found: {file_path}')
            if not os.access(file_path, os.R_OK):
                return self._create_error_result(f'Permission denied: {file_path}')
                
            # 지원되는 파일인지 확인
            if not self._is_supported_file(file_path):
                return self._create_error_result(f'Unsupported file type: {file_path}')
                
            # 파일 크기 확인
            try:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    return self._create_error_result(f'Empty file: {file_path}')
                if file_size > self._max_file_size:
                    return self._create_error_result(f'File too large: {file_path}')
            except Exception as e:
                return self._create_error_result(f'Error checking file size: {e}')
                
            # 언어 설정
            if not language:
                language = self._get_file_language(file_path)
                
            # 분석 시도
            analysis_results: List[AnalysisResult] = []
            analysis_priority = self._get_analysis_priority(file_path)
            
            for attempt in range(self._retry_count):
                try:
                    # 우선순위에 따라 분석 시도
                    for method in analysis_priority:
                        result = await self._analyze_with_method(file_path, method, language)
                        if result and self._validate_analysis_result(result):
                            analysis_results.append(result)
                            
                    # 분석 결과가 있으면 병합하여 반환
                    if analysis_results:
                        merged_result = self._merge_analysis_results(analysis_results)
                        await self._cache_result(file_path, merged_result)
                        return merged_result
                        
                    if attempt < self._retry_count - 1:
                        await asyncio.sleep(self._retry_delay * (attempt + 1))
                        continue
                        
                    return self._create_error_result('Analysis failed for all supported methods')
                except Exception as e:
                    if attempt < self._retry_count - 1:
                        await asyncio.sleep(self._retry_delay * (attempt + 1))
                        continue
                    logger.error(f"파일 분석 실패: {file_path}: {e}")
                    return self._create_error_result(str(e))
                    
    async def analyze_files(self, files: List[Tuple[str, str]]) -> List[AnalysisResult]:
        """여러 파일 분석 (병렬 처리)"""
        # 작업자 시작
        await self.start_workers()
        
        try:
            # 작업 큐에 파일 추가
            for file_path, language in files:
                await self.analysis_queue.put((file_path, language))
                
            # 결과 수집
            analysis_results: List[AnalysisResult] = []
            for _ in range(len(files)):
                queue_result = await self.result_queue.get()
                if isinstance(queue_result, dict):
                    analysis_results.append(queue_result)
                else:
                    analysis_results.append(self._create_error_result('Invalid result type'))
                    
            return analysis_results
        finally:
            # 작업자 중지
            await self.stop_workers()
            
    async def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 작업자 중지
            await self.stop_workers()
            
            # 스레드 풀 종료
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            
            # 캐시 정리
            async with self._cache_lock:
                self.cache.clear()
                
            # 파일 락 정리
            self._file_locks.clear()
            
            # 큐 정리
            while not self.analysis_queue.empty():
                try:
                    self.analysis_queue.get_nowait()
                    self.analysis_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                    
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            # 메모리 정리
            import gc
            gc.collect()
        except Exception as e:
            logger.error(f"리소스 정리 실패: {e}")
            
    async def start_workers(self, num_workers: Optional[int] = None) -> None:
        """작업자 프로세스 시작"""
        if num_workers is None:
            num_workers = os.cpu_count() or 4
            
        for _ in range(num_workers):
            task = asyncio.create_task(self._worker())
            self.worker_tasks.append(task)
            
    async def stop_workers(self) -> None:
        """작업자 프로세스 중지"""
        self.stop_event.set()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        self.stop_event.clear()
        
    async def _worker(self) -> None:
        """작업자 프로세스"""
        while not self.stop_event.is_set():
            try:
                # 작업 큐에서 파일 가져오기
                file_path, language = await self.analysis_queue.get()
                
                # 동시 분석 제한
                async with self._analysis_semaphore:
                    try:
                        # 파일 분석
                        result = await self.analyze_file(file_path, language)
                        
                        # 결과 검증
                        if not self._validate_analysis_result(result):
                            result = self._create_error_result('Invalid analysis result structure')
                            
                        await self.result_queue.put(result)
                    except Exception as e:
                        logger.error(f"작업자 분석 실패: {file_path}: {e}")
                        await self.result_queue.put(self._create_error_result(str(e)))
                    finally:
                        self.analysis_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"작업자 오류: {e}")
                continue
                
    def _generate_file_hash(self, file_path: str) -> str:
        """파일 해시 생성 (메모리 효율적)"""
        try:
            stat = os.stat(file_path)
            return f"{stat.st_mtime}:{stat.st_size}"
        except Exception as e:
            logger.error(f"Error stat file for hash: {file_path}: {e}")
            return ''
            
    @lru_cache(maxsize=1000)
    def _analyze_python_file(self, content: str) -> AnalysisResult:
        """Python 파일 분석 (캐시 적용)"""
        try:
            # 파일 내용 검증
            if not content.strip():
                return {'error': 'Empty file content'}
                
            # AST 파싱
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return {'error': f'Syntax error: {str(e)}'}
            except Exception as e:
                return {'error': f'Parse error: {str(e)}'}
                
            # AST 분석
            analyzer = PythonASTAnalyzer()
            analyzer.visit(tree)
            result = analyzer.get_result()
            
            # 결과 검증
            if not isinstance(result, dict):
                return {'error': 'Invalid analysis result type'}
            if not any(key in result for key in ['functions', 'classes', 'imports', 'variables', 'calls']):
                return {'error': 'Invalid analysis result structure'}
                
            return result
        except Exception as e:
            logger.error(f"Python AST 분석 실패: {e}")
            return {'error': str(e)}

class PythonASTAnalyzer(ast.NodeVisitor):
    """Python AST 분석기"""
    
    def __init__(self) -> None:
        self.result: AnalysisResult = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'calls': [],
            'language': 'python',
            'dependencies': [],
            'metrics': asdict(CodeMetrics())
        }
        
    def visit_FunctionDef(self, node: FunctionNode) -> None:
        """함수 정의 방문"""
        function_info: Dict[str, Any] = {
            'name': node.name,
            'parameters': [arg.arg for arg in node.args.args],
            'start_line': node.lineno,
            'end_line': getattr(node, 'end_lineno', node.lineno),
            'content': ast.unparse(node),
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [ast.unparse(d) for d in node.decorator_list],
            'calls': []
        }
        
        # 함수 내용 분석
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    function_info['calls'].append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    function_info['calls'].append(ast.unparse(child.func))
                    
            # 부수 효과 감지
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('_'):
                        function_info['side_effects'].append(f"Modifies global variable: {target.id}")
                        
        self.result['functions'].append(function_info)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """클래스 정의 방문"""
        class_info: Dict[str, Any] = {
            'name': node.name,
            'methods': [],
            'attributes': [],
            'inheritance': [ast.unparse(base) for base in node.bases],
            'start_line': node.lineno,
            'end_line': getattr(node, 'end_lineno', node.lineno),
            'content': ast.unparse(node),
            'is_abstract': any(isinstance(d, ast.Name) and d.id == 'abstractmethod' 
                             for d in node.decorator_list)
        }
        
        # 클래스 내용 분석
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                class_info['methods'].append(child.name)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        class_info['attributes'].append(target.id)
                        
        self.result['classes'].append(class_info)
        self.generic_visit(node)
        
    def visit_Import(self, node: ast.Import) -> None:
        """임포트 방문"""
        for name in node.names:
            import_info: Dict[str, Any] = {
                'module': name.name,
                'items': [name.name],
                'alias': name.asname,
                'start_line': node.lineno,
                'end_line': getattr(node, 'end_lineno', node.lineno),
                'is_from_import': False
            }
            self.result['imports'].append(import_info)
            
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """from 임포트 방문"""
        for name in node.names:
            import_info: Dict[str, Any] = {
                'module': node.module or '',
                'items': [name.name],
                'alias': name.asname,
                'start_line': node.lineno,
                'end_line': getattr(node, 'end_lineno', node.lineno),
                'is_from_import': True
            }
            self.result['imports'].append(import_info)
            
    def visit_Call(self, node: ast.Call) -> None:
        """함수 호출 방문"""
        call_info: Optional[Dict[str, Any]] = None
        if isinstance(node.func, ast.Name):
            call_info = {
                'name': node.func.id,
                'arguments': [ast.unparse(arg) for arg in node.args],
                'start_line': node.lineno,
                'end_line': getattr(node, 'end_lineno', node.lineno)
            }
        elif isinstance(node.func, ast.Attribute):
            call_info = {
                'name': ast.unparse(node.func),
                'arguments': [ast.unparse(arg) for arg in node.args],
                'start_line': node.lineno,
                'end_line': getattr(node, 'end_lineno', node.lineno)
            }
        if call_info:
            self.result['calls'].append(call_info)
            
    def visit_Assign(self, node: ast.Assign) -> None:
        """변수 할당 방문"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                variable_info: Dict[str, Any] = {
                    'name': target.id,
                    'value': ast.unparse(node.value),
                    'start_line': node.lineno,
                    'end_line': getattr(node, 'end_lineno', node.lineno)
                }
                self.result['variables'].append(variable_info)

    def get_result(self) -> Dict[str, Any]:
        return dict(self.result)

def get_code_analyzer() -> CodeAnalyzer:
    """코드 분석기 인스턴스 반환"""
    return CodeAnalyzer()