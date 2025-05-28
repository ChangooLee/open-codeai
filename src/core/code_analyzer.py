"""
Open CodeAI - 코드 분석기 구현
Tree-sitter를 사용한 코드 파싱 및 분석
"""
import ast
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

from ..config import settings
from ..utils.logger import get_logger, log_performance

logger = get_logger(__name__)

class CodeLanguage(Enum):
    """지원하는 프로그래밍 언어"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SCALA = "scala"
    KOTLIN = "kotlin"

@dataclass
class FunctionInfo:
    """함수 정보"""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    complexity: int = 1
    is_async: bool = False
    decorators: List[str] = None

@dataclass
class ClassInfo:
    """클래스 정보"""
    name: str
    start_line: int
    end_line: int
    methods: List[FunctionInfo]
    attributes: List[str]
    inheritance: List[str] = None
    docstring: Optional[str] = None

@dataclass
class ImportInfo:
    """임포트 정보"""
    module: str
    alias: Optional[str] = None
    items: List[str] = None
    is_from_import: bool = False

@dataclass
class CodeAnalysisResult:
    """코드 분석 결과"""
    language: str
    file_path: Optional[str] = None
    
    # 구조 정보
    functions: List[FunctionInfo] = None
    classes: List[ClassInfo] = None
    imports: List[ImportInfo] = None
    
    # 메트릭
    lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    complexity: int = 0
    
    # 품질 지표
    test_coverage_estimate: float = 0.0
    maintainability_index: float = 0.0
    code_smells: List[str] = None
    
    # 의존성
    dependencies: Set[str] = None
    exports: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = asdict(self)
        # Set을 list로 변환
        if result.get('dependencies'):
            result['dependencies'] = list(result['dependencies'])
        return result

class CodeAnalyzer:
    """
    코드 분석기
    
    - Tree-sitter를 사용한 구문 분석
    - AST 기반 메트릭 계산
    - 코드 품질 평가
    - 의존성 분석
    """
    
    def __init__(self):
        self.parsers = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Tree-sitter 언어 파서 초기화
        if HAS_TREE_SITTER:
            self._initialize_parsers()
        else:
            logger.warning("Tree-sitter가 설치되지 않았습니다. 기본 분석만 사용됩니다.")
    
    def _initialize_parsers(self):
        """Tree-sitter 파서들 초기화"""
        try:
            # 언어별 파서 설정 (실제로는 빌드된 언어 파일이 필요)
            language_configs = {
                CodeLanguage.PYTHON: "tree-sitter-python",
                CodeLanguage.JAVASCRIPT: "tree-sitter-javascript", 
                CodeLanguage.TYPESCRIPT: "tree-sitter-typescript",
                CodeLanguage.JAVA: "tree-sitter-java",
                CodeLanguage.CPP: "tree-sitter-cpp",
                CodeLanguage.C: "tree-sitter-c",
                CodeLanguage.GO: "tree-sitter-go",
                CodeLanguage.RUST: "tree-sitter-rust"
            }
            
            for lang, lib_name in language_configs.items():
                try:
                    # 실제 구현에서는 빌드된 .so 파일을 로드해야 함
                    # 여기서는 기본 파서로 대체
                    parser = Parser()
                    self.parsers[lang] = parser
                    logger.info(f"{lang.value} 파서 초기화 완료")
                except Exception as e:
                    logger.warning(f"{lang.value} 파서 초기화 실패: {e}")
                    
        except Exception as e:
            logger.error(f"Tree-sitter 파서 초기화 실패: {e}")
    
    async def analyze_code(
        self, 
        code: str, 
        language: str = "python",
        file_path: Optional[str] = None
    ) -> CodeAnalysisResult:
        """
        코드 분석 실행
        
        Args:
            code: 분석할 코드
            language: 프로그래밍 언어
            file_path: 파일 경로 (선택적)
            
        Returns:
            분석 결과
        """
        
        try:
            lang_enum = CodeLanguage(language)
        except ValueError:
            logger.warning(f"지원하지 않는 언어: {language}")
            lang_enum = CodeLanguage.PYTHON
        
        logger.info(f"코드 분석 시작 - 언어: {language}, 길이: {len(code)} chars")
        
        # 비동기 분석 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._analyze_code_sync,
            code,
            lang_enum,
            file_path
        )
        
        logger.info(f"코드 분석 완료 - 함수: {len(result.functions or [])}, 클래스: {len(result.classes or [])}")
        return result
    
    @log_performance("code_analysis")
    def _analyze_code_sync(
        self,
        code: str,
        language: CodeLanguage,
        file_path: Optional[str]
    ) -> CodeAnalysisResult:
        """동기 코드 분석"""
        
        result = CodeAnalysisResult(
            language=language.value,
            file_path=file_path,
            functions=[],
            classes=[],
            imports=[],
            code_smells=[],
            dependencies=set()
        )
        
        # 기본 메트릭 계산
        result.lines_of_code, result.comment_lines, result.blank_lines = self._count_lines(code)
        
        # 언어별 분석
        if language == CodeLanguage.PYTHON:
            self._analyze_python(code, result)
        elif language == CodeLanguage.JAVASCRIPT:
            self._analyze_javascript(code, result)
        elif language == CodeLanguage.TYPESCRIPT:
            self._analyze_typescript(code, result)
        else:
            # 기본 분석 (패턴 매칭 기반)
            self._analyze_generic(code, result)
        
        # 품질 지표 계산
        result.complexity = self._calculate_complexity(result)
        result.maintainability_index = self._calculate_maintainability(result)
        result.test_coverage_estimate = self._estimate_test_coverage(code, result)
        
        return result
    
    def _count_lines(self, code: str) -> Tuple[int, int, int]:
        """라인 수 계산 (코드, 주석, 빈 줄)"""
        lines = code.split('\n')
        
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                comment_lines += 1
            else:
                code_lines += 1
        
        return code_lines, comment_lines, blank_lines
    
    def _analyze_python(self, code: str, result: CodeAnalysisResult):
        """Python 코드 분석"""
        try:
            tree = ast.parse(code)
            
            # AST 방문자로 분석
            visitor = PythonASTVisitor(result)
            visitor.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Python 구문 오류: {e}")
            result.code_smells.append(f"구문 오류: {e}")
        except Exception as e:
            logger.error(f"Python 분석 실패: {e}")
    
    def _analyze_javascript(self, code: str, result: CodeAnalysisResult):
        """JavaScript 코드 분석 (패턴 매칭 기반)"""
        
        # 함수 찾기
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{'
        matches = re.finditer(function_pattern, code)
        
        for match in matches:
            func_name = match.group(1)
            start_line = code[:match.start()].count('\n') + 1
            
            func_info = FunctionInfo(
                name=func_name,
                start_line=start_line,
                end_line=start_line + 10,  # 추정
                parameters=[]
            )
            result.functions.append(func_info)
        
        # 임포트 찾기
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        matches = re.finditer(import_pattern, code)
        
        for match in matches:
            module = match.group(1)
            import_info = ImportInfo(
                module=module,
                is_from_import=True
            )
            result.imports.append(import_info)
            result.dependencies.add(module)
    
    def _analyze_typescript(self, code: str, result: CodeAnalysisResult):
        """TypeScript 코드 분석"""
        # JavaScript 분석과 유사하지만 타입 정보 추가
        self._analyze_javascript(code, result)
        
        # 인터페이스 찾기
        interface_pattern = r'interface\s+(\w+)\s*\{'
        matches = re.finditer(interface_pattern, code)
        
        for match in matches:
            interface_name = match.group(1)
            start_line = code[:match.start()].count('\n') + 1
            
            # 인터페이스를 클래스로 처리
            class_info = ClassInfo(
                name=interface_name,
                start_line=start_line,
                end_line=start_line + 5,
                methods=[],
                attributes=[]
            )
            result.classes.append(class_info)
    
    def _analyze_generic(self, code: str, result: CodeAnalysisResult):
        """일반적인 코드 분석 (패턴 매칭)"""
        
        # 함수 패턴들
        function_patterns = [
            r'def\s+(\w+)\s*\(',  # Python
            r'function\s+(\w+)\s*\(',  # JavaScript
            r'(\w+)\s*\([^)]*\)\s*\{',  # C/Java style
        ]
        
        for pattern in function_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                func_name = match.group(1)
                start_line = code[:match.start()].count('\n') + 1
                
                func_info = FunctionInfo(
                    name=func_name,
                    start_line=start_line,
                    end_line=start_line + 1,
                    parameters=[]
                )
                result.functions.append(func_info)
    
    def _calculate_complexity(self, result: CodeAnalysisResult) -> int:
        """코드 복잡도 계산"""
        complexity = 1  # 기본 복잡도
        
        # 함수당 복잡도 추가
        for func in result.functions or []:
            complexity += func.complexity
        
        # 클래스당 복잡도 추가  
        for cls in result.classes or []:
            complexity += len(cls.methods) * 2
        
        return complexity
    
    def _calculate_maintainability(self, result: CodeAnalysisResult) -> float:
        """유지보수성 지수 계산"""
        
        if result.lines_of_code == 0:
            return 100.0
        
        # 간단한 유지보수성 지수 계산
        comment_ratio = result.comment_lines / result.lines_of_code if result.lines_of_code > 0 else 0
        complexity_penalty = min(result.complexity / 10, 1.0)
        
        maintainability = 100 - (complexity_penalty * 50) + (comment_ratio * 20)
        return max(0.0, min(100.0, maintainability))
    
    def _estimate_test_coverage(self, code: str, result: CodeAnalysisResult) -> float:
        """테스트 커버리지 추정"""
        
        # 테스트 관련 키워드 찾기
        test_keywords = ['test_', 'test ', 'Test', 'describe', 'it(', '@Test']
        test_score = 0
        
        for keyword in test_keywords:
            test_score += code.count(keyword) * 10
        
        # 함수 대비 테스트 비율 추정
        function_count = len(result.functions or [])
        if function_count > 0:
            estimated_coverage = min(test_score / function_count, 100.0)
        else:
            estimated_coverage = 0.0
        
        return estimated_coverage
    
    async def analyze_file(self, file_path: str) -> Optional[CodeAnalysisResult]:
        """파일 분석"""
        
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"파일을 찾을 수 없습니다: {file_path}")
                return None
            
            # 언어 감지
            language = self._detect_language(path)
            
            # 파일 읽기
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # 분석 실행
            result = await self.analyze_code(code, language, file_path)
            return result
            
        except Exception as e:
            logger.error(f"파일 분석 실패 {file_path}: {e}")
            return None
    
    def _detect_language(self, file_path: Path) -> str:
        """파일 확장자로 언어 감지"""
        
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.scala': 'scala',
            '.kt': 'kotlin'
        }
        
        return extension_map.get(file_path.suffix.lower(), 'python')
    
    async def analyze_directory(
        self, 
        directory_path: str,
        max_files: int = 1000
    ) -> List[CodeAnalysisResult]:
        """디렉토리 전체 분석"""
        
        try:
            path = Path(directory_path)
            if not path.exists():
                logger.error(f"디렉토리를 찾을 수 없습니다: {directory_path}")
                return []
            
            # 지원하는 파일 확장자
            supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb'}
            
            # 파일 목록 수집
            files = []
            for ext in supported_extensions:
                files.extend(list(path.rglob(f'*{ext}'))[:max_files])
            
            logger.info(f"디렉토리 분석 완료: {len(valid_results)} 파일 성공")
            return valid_results
            
        except Exception as e:
            logger.error(f"디렉토리 분석 실패 {directory_path}: {e}")
            return []

class PythonASTVisitor(ast.NodeVisitor):
    """Python AST 방문자"""
    
    def __init__(self, result: CodeAnalysisResult):
        self.result = result
        self.current_class = None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """함수 정의 방문"""
        
        # 파라미터 추출
        parameters = [arg.arg for arg in node.args.args]
        
        # 반환 타입 추출
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # 독스트링 추출
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # 데코레이터 추출
        decorators = [ast.unparse(dec) if hasattr(ast, 'unparse') else str(dec) 
                     for dec in node.decorator_list]
        
        # 복잡도 계산 (간단한 버전)
        complexity = self._calculate_function_complexity(node)
        
        func_info = FunctionInfo(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators
        )
        
        if self.current_class:
            self.current_class.methods.append(func_info)
        else:
            self.result.functions.append(func_info)
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """비동기 함수 정의 방문"""
        self.visit_FunctionDef(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """클래스 정의 방문"""
        
        # 상속 정보 추출
        inheritance = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inheritance.append(base.id)
            elif isinstance(base, ast.Attribute):
                inheritance.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else str(base.attr))
        
        # 독스트링 추출
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # 속성 추출 (간단한 버전)
        attributes = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        class_info = ClassInfo(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            methods=[],
            attributes=attributes,
            inheritance=inheritance,
            docstring=docstring
        )
        
        # 현재 클래스 설정하고 메서드들 방문
        old_class = self.current_class
        self.current_class = class_info
        self.generic_visit(node)
        self.current_class = old_class
        
        self.result.classes.append(class_info)
    
    def visit_Import(self, node: ast.Import):
        """임포트 방문"""
        for alias in node.names:
            import_info = ImportInfo(
                module=alias.name,
                alias=alias.asname,
                is_from_import=False
            )
            self.result.imports.append(import_info)
            self.result.dependencies.add(alias.name.split('.')[0])
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """from 임포트 방문"""
        if node.module:
            items = [alias.name for alias in node.names]
            import_info = ImportInfo(
                module=node.module,
                items=items,
                is_from_import=True
            )
            self.result.imports.append(import_info)
            self.result.dependencies.add(node.module.split('.')[0])
        
        self.generic_visit(node)
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """함수의 순환 복잡도 계산"""
        complexity = 1  # 기본 복잡도
        
        for child in ast.walk(node):
            # 분기문들
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            # 예외 처리
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            # 논리 연산자
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

# 코드 품질 검사기
class CodeQualityChecker:
    """코드 품질 검사"""
    
    @staticmethod
    def check_python_smells(code: str) -> List[str]:
        """Python 코드 스멜 검사"""
        smells = []
        
        # 긴 함수 검사
        if 'def ' in code:
            functions = re.findall(r'def\s+\w+.*?(?=\ndef|\nclass|\Z)', code, re.DOTALL)
            for func in functions:
                lines = len(func.split('\n'))
                if lines > 50:
                    smells.append(f"긴 함수 발견 ({lines} 줄)")
        
        # 깊은 중첩 검사
        max_indent = 0
        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            if indent > max_indent:
                max_indent = indent
        
        if max_indent > 16:  # 4칸 들여쓰기 기준 4단계 초과
            smells.append(f"과도한 중첩 ({max_indent//4} 단계)")
        
        # 매직 넘버 검사
        magic_numbers = re.findall(r'\b(?!0|1)\d{2,}\b', code)
        if len(magic_numbers) > 5:
            smells.append(f"매직 넘버 과다 사용 ({len(magic_numbers)}개)")
        
        # 긴 파라미터 목록 검사
        long_params = re.findall(r'def\s+\w+\([^)]{50,}\)', code)
        if long_params:
            smells.append(f"긴 파라미터 목록 ({len(long_params)}개 함수)")
        
        return smells

# 의존성 분석기
class DependencyAnalyzer:
    """의존성 분석기"""
    
    @staticmethod
    def analyze_python_dependencies(code: str) -> Dict[str, Any]:
        """Python 의존성 분석"""
        dependencies = {
            'stdlib': set(),
            'third_party': set(),
            'local': set()
        }
        
        # 표준 라이브러리 목록 (일부)
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'collections', 
            'itertools', 'functools', 'pathlib', 're', 'math', 'random',
            'urllib', 'http', 'asyncio', 'threading', 'multiprocessing'
        }
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module in stdlib_modules:
                            dependencies['stdlib'].add(module)
                        elif module.startswith('.'):
                            dependencies['local'].add(module)
                        else:
                            dependencies['third_party'].add(module)
                
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split('.')[0]
                    if node.level > 0:  # 상대 임포트
                        dependencies['local'].add(node.module)
                    elif module in stdlib_modules:
                        dependencies['stdlib'].add(module)
                    else:
                        dependencies['third_party'].add(module)
        
        except SyntaxError:
            pass
        
        return {k: list(v) for k, v in dependencies.items()}

# 코드 메트릭 계산기
class CodeMetrics:
    """코드 메트릭 계산"""
    
    @staticmethod
    def calculate_halstead_metrics(code: str) -> Dict[str, float]:
        """Halstead 복잡도 메트릭 계산"""
        
        # 연산자와 피연산자 패턴 (간단한 버전)
        operators = re.findall(r'[+\-*/=<>!&|^%]|==|!=|<=|>=|and|or|not|in|is', code)
        operands = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+\b', code)
        
        # 고유 연산자/피연산자 수
        unique_operators = len(set(operators))
        unique_operands = len(set(operands))
        
        # 전체 연산자/피연산자 수
        total_operators = len(operators)
        total_operands = len(operands)
        
        # Halstead 메트릭
        vocabulary = unique_operators + unique_operands
        length = total_operators + total_operands
        
        if unique_operators > 0 and unique_operands > 0:
            volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
            difficulty = (unique_operators / 2) * (total_operands / unique_operands)
            effort = difficulty * volume
        else:
            volume = difficulty = effort = 0
        
        return {
            'vocabulary': vocabulary,
            'length': length,
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }
    
    @staticmethod
    def calculate_maintainability_index(
        lines_of_code: int,
        complexity: int,
        halstead_volume: float
    ) -> float:
        """유지보수성 지수 계산 (Microsoft 공식)"""
        
        if lines_of_code == 0:
            return 100.0
        
        import math
        
        # MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        try:
            mi = (171 - 
                  5.2 * math.log(max(halstead_volume, 1)) - 
                  0.23 * complexity - 
                  16.2 * math.log(max(lines_of_code, 1)))
            
            # 0-100 범위로 정규화
            return max(0, min(100, mi))
            
        except (ValueError, ZeroDivisionError):
            return 50.0  # 기본값

# 전역 코드 분석기 인스턴스
_code_analyzer_instance = None

def get_code_analyzer() -> CodeAnalyzer:
    """전역 코드 분석기 인스턴스 반환"""
    global _code_analyzer_instance
    
    if _code_analyzer_instance is None:
        _code_analyzer_instance = CodeAnalyzer()
    
    return _code_analyzer_instance분석 시작: {len(files)} 파일")
            
            # 병렬 분석
            tasks = [self.analyze_file(str(file_path)) for file_path in files]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 성공한 결과만 반환
            valid_results = [r for r in results if isinstance(r, CodeAnalysisResult)]
            
            logger.info(f"디렉토리