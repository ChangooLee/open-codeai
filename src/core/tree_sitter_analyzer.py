"""
Tree-sitter 기반 코드 분석기
"""
import os
from typing import Dict, List, Any, Optional, Set, Tuple, cast, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class Language(Enum):
    """지원하는 언어"""
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
    CSS = "css"
    VUE = "vue"
    HTML = "html"
    ANGULAR = "angular"

@dataclass
class AnalysisResult:
    """분석 결과"""
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    variables: List[Dict[str, Any]] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    language: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

class TreeSitterAnalyzer:
    """Tree-sitter 기반 코드 분석기"""
    
    def __init__(self) -> None:
        self.parsers: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialize_parsers()
        
    def _initialize_parsers(self) -> None:
        """Tree-sitter 파서 초기화"""
        try:
            from tree_sitter import Language as TreeSitterLanguage, Parser
            
            # 언어별 파서 로드
            for lang in list(Language):
                try:
                    parser = Parser()
                    # Tree-sitter 언어 파일 경로 설정
                    language_path = os.path.join('build', 'my-languages.so')
                    if not os.path.exists(language_path):
                        logger.warning(f"Tree-sitter language file not found: {language_path}")
                        continue
                        
                    # 언어별 파서 설정
                    try:
                        # Tree-sitter Language 클래스 초기화
                        language = cast(Any, TreeSitterLanguage)(language_path, lang.value)
                        parser.language = language
                        self.parsers[lang.value] = parser
                    except Exception as e:
                        logger.warning(f"Failed to load Tree-sitter language {lang.value}: {e}")
                        
                except Exception as e:
                    logger.warning(f"Tree-sitter parser for {lang.value} not available: {e}")
                    
        except ImportError:
            logger.warning("Tree-sitter not available")
            
    def detect_language(self, file_path: str) -> Optional[Language]:
        """파일 확장자로 언어 감지"""
        extension = Path(file_path).suffix.lower()
        extension_map = {
            '.py': Language.PYTHON,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.JAVASCRIPT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.TYPESCRIPT,
            '.mjs': Language.JAVASCRIPT,
            '.java': Language.JAVA,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.cxx': Language.CPP,
            '.c': Language.C,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.php': Language.PHP,
            '.rb': Language.RUBY,
            '.css': Language.CSS,
            '.vue': Language.VUE,
            '.html': Language.HTML,
            '.htm': Language.HTML,
        }
        
        # Angular: .ts + @Component decorator or .html with Angular template syntax
        if extension == '.ts':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if '@Component' in content or '@NgModule' in content:
                    return Language.ANGULAR
            except Exception:
                pass
                
        return extension_map.get(extension)
        
    async def analyze(self, file_path: str, language: str) -> Dict[str, Any]:
        """파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            result = self._analyze_content(content, language)
            if result:
                return {
                    'functions': result.functions,
                    'classes': result.classes,
                    'imports': result.imports,
                    'variables': result.variables,
                    'calls': result.calls,
                    'language': result.language,
                    'dependencies': [],
                    'metrics': result.metrics,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
            return {'error': 'Analysis failed'}
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {'error': str(e)}
            
    def _analyze_content(self, content: str, language: str) -> Optional[AnalysisResult]:
        """내용 분석"""
        try:
            lang = Language(language)
            if lang.value not in self.parsers:
                return None
                
            parser = self.parsers[lang.value]
            tree = parser.parse(bytes(content, 'utf8'))
            
            result = AnalysisResult(language=language)
            
            # 기본 분석
            self._analyze_tree(tree.root_node, result)
            
            # 언어별 특수 분석
            if lang == Language.PYTHON:
                self._analyze_python_specific(content, result)
            elif lang in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                self._analyze_js_specific(content, result)
            elif lang == Language.JAVA:
                self._analyze_java_specific(content, result)
                
            # 메트릭 계산
            self._calculate_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return None
            
    def _analyze_tree(self, node: Any, result: AnalysisResult) -> None:
        """트리 분석"""
        try:
            if node.type == 'function_definition':
                result.functions.append(self._analyze_function(node))
            elif node.type == 'class_definition':
                result.classes.append(self._analyze_class(node))
            elif node.type == 'import_statement':
                result.imports.append(self._analyze_import(node))
            elif node.type == 'call_expression':
                result.calls.append(self._analyze_call(node))
            elif node.type == 'variable_declaration':
                result.variables.append(self._analyze_variable(node))
                
            for child in node.children:
                self._analyze_tree(child, result)
                
        except Exception as e:
            result.errors.append(f"Error analyzing node {node.type}: {str(e)}")
            
    def _analyze_function(self, node: Any) -> Dict[str, Any]:
        """함수 분석"""
        try:
            name_node = node.child_by_field_name('name')
            params_node = node.child_by_field_name('parameters')
            body_node = node.child_by_field_name('body')
            
            return {
                'name': self._get_node_text(name_node),
                'parameters': self._get_parameters(params_node),
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'content': self._get_node_text(node),
                'complexity': self._calculate_complexity(body_node),
                'is_async': 'async' in self._get_node_text(node),
                'decorators': self._get_decorators(node),
                'calls': self._get_function_calls(body_node)
            }
        except Exception as e:
            logger.error(f"Error analyzing function: {e}")
            return {}
            
    def _analyze_class(self, node: Any) -> Dict[str, Any]:
        """클래스 분석"""
        try:
            name_node = node.child_by_field_name('name')
            body_node = node.child_by_field_name('body')
            
            return {
                'name': self._get_node_text(name_node),
                'methods': self._get_class_methods(body_node),
                'attributes': self._get_class_attributes(body_node),
                'inheritance': self._get_class_inheritance(node),
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'content': self._get_node_text(node),
                'is_abstract': self._is_abstract_class(node)
            }
        except Exception as e:
            logger.error(f"Error analyzing class: {e}")
            return {}
            
    def _analyze_import(self, node: Any) -> Dict[str, Any]:
        """임포트 분석"""
        try:
            return {
                'module': self._get_node_text(node.child_by_field_name('module')),
                'items': [self._get_node_text(item) for item in node.children_by_field_name('name')],
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'is_from_import': node.type == 'from_import_statement'
            }
        except Exception as e:
            logger.error(f"Error analyzing import: {e}")
            return {}
            
    def _analyze_call(self, node: Any) -> Dict[str, Any]:
        """함수 호출 분석"""
        try:
            return {
                'name': self._get_node_text(node.child_by_field_name('function')),
                'arguments': [self._get_node_text(arg) for arg in node.children_by_field_name('arguments')],
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1
            }
        except Exception as e:
            logger.error(f"Error analyzing call: {e}")
            return {}
            
    def _analyze_variable(self, node: Any) -> Dict[str, Any]:
        """변수 분석"""
        try:
            return {
                'name': self._get_node_text(node.child_by_field_name('name')),
                'type': self._get_node_text(node.child_by_field_name('type')),
                'value': self._get_node_text(node.child_by_field_name('value')),
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1
            }
        except Exception as e:
            logger.error(f"Error analyzing variable: {e}")
            return {}
            
    def _analyze_python_specific(self, content: str, result: AnalysisResult) -> None:
        """Python 특수 분석"""
        try:
            # 데코레이터 분석
            decorator_pattern = r'@(\w+)'
            decorators = re.findall(decorator_pattern, content)
            result.metrics['decorators'] = list(set(decorators))
            
            # 타입 힌트 분석
            type_hint_pattern = r':\s*([A-Za-z_][A-Za-z0-9_]*(\[[^\]]+\])?)'
            type_hints = re.findall(type_hint_pattern, content)
            result.metrics['type_hints'] = list(set(hint[0] for hint in type_hints))
            
        except Exception as e:
            result.errors.append(f"Error in Python specific analysis: {str(e)}")
            
    def _analyze_js_specific(self, content: str, result: AnalysisResult) -> None:
        """JavaScript/TypeScript 특수 분석"""
        try:
            # ES6+ 기능 분석
            es6_features = {
                'arrow_functions': len(re.findall(r'=>', content)),
                'template_literals': len(re.findall(r'`[^`]*`', content)),
                'destructuring': len(re.findall(r'const\s*\{[^}]*\}', content)),
                'spread_operator': len(re.findall(r'\.\.\.', content))
            }
            result.metrics['es6_features'] = es6_features
            
            # React/Vue 컴포넌트 분석
            if 'React' in content or 'react' in content:
                result.metrics['framework'] = 'react'
            elif 'Vue' in content or 'vue' in content:
                result.metrics['framework'] = 'vue'
                
        except Exception as e:
            result.errors.append(f"Error in JS specific analysis: {str(e)}")
            
    def _analyze_java_specific(self, content: str, result: AnalysisResult) -> None:
        """Java 특수 분석"""
        try:
            # 어노테이션 분석
            annotation_pattern = r'@(\w+)'
            annotations = re.findall(annotation_pattern, content)
            result.metrics['annotations'] = list(set(annotations))
            
            # 제네릭 타입 분석
            generic_pattern = r'<([^>]+)>'
            generics = re.findall(generic_pattern, content)
            result.metrics['generics'] = list(set(generics))
            
        except Exception as e:
            result.errors.append(f"Error in Java specific analysis: {str(e)}")
            
    def _calculate_metrics(self, result: AnalysisResult) -> None:
        """메트릭 계산"""
        try:
            # 기본 메트릭
            result.metrics.update({
                'total_functions': len(result.functions),
                'total_classes': len(result.classes),
                'total_imports': len(result.imports),
                'total_variables': len(result.variables),
                'total_calls': len(result.calls)
            })
            
            # 복잡도 메트릭
            if result.functions:
                complexities = [f.get('complexity', 1) for f in result.functions]
                result.metrics.update({
                    'avg_complexity': sum(complexities) / len(complexities),
                    'max_complexity': max(complexities),
                    'min_complexity': min(complexities)
                })
                
            # 코드 품질 메트릭
            result.metrics.update({
                'has_docstrings': any(f.get('docstring') for f in result.functions),
                'has_type_hints': any(f.get('return_type') for f in result.functions),
                'has_tests': any('test' in f.get('name', '').lower() for f in result.functions)
            })
            
        except Exception as e:
            result.errors.append(f"Error calculating metrics: {str(e)}")
            
    def _get_node_text(self, node: Any) -> str:
        """노드의 텍스트 추출"""
        if not node:
            return ""
        return cast(str, node.text.decode('utf8'))
        
    def _get_parameters(self, node: Any) -> List[str]:
        """함수 파라미터 추출"""
        params = []
        if node:
            for param in node.children:
                if param.type == 'identifier':
                    params.append(self._get_node_text(param))
        return params
        
    def _get_class_methods(self, node: Any) -> List[Dict[str, Any]]:
        """클래스 메서드 추출"""
        methods = []
        if node:
            for child in node.children:
                if child.type == 'function_definition':
                    methods.append(self._analyze_function(child))
        return methods
        
    def _get_class_attributes(self, node: Any) -> List[str]:
        """클래스 속성 추출"""
        attributes = []
        if node:
            for child in node.children:
                if child.type == 'variable_declaration':
                    name_node = child.child_by_field_name('name')
                    if name_node:
                        attributes.append(self._get_node_text(name_node))
        return attributes
        
    def _get_class_inheritance(self, node: Any) -> List[str]:
        """클래스 상속 추출"""
        inheritance = []
        bases_node = node.child_by_field_name('bases')
        if bases_node:
            for base in bases_node.children:
                if base.type == 'identifier':
                    inheritance.append(self._get_node_text(base))
        return inheritance
        
    def _is_abstract_class(self, node: Any) -> bool:
        """추상 클래스 여부 확인"""
        decorators = self._get_decorators(node)
        return any(d == 'abstract' for d in decorators)
        
    def _get_decorators(self, node: Any) -> List[str]:
        """데코레이터 추출"""
        decorators = []
        decorator_list = node.child_by_field_name('decorator_list')
        if decorator_list:
            for decorator in decorator_list.children:
                if decorator.type == 'identifier':
                    decorators.append(self._get_node_text(decorator))
        return decorators
        
    def _get_function_calls(self, node: Any) -> List[str]:
        """함수 호출 추출"""
        calls = []
        if node:
            for child in node.children:
                if child.type == 'call_expression':
                    func_node = child.child_by_field_name('function')
                    if func_node:
                        calls.append(self._get_node_text(func_node))
        return calls
        
    def _calculate_complexity(self, node: Any) -> int:
        """복잡도 계산"""
        complexity = 1
        if node:
            for child in node.children:
                if child.type in ['if_statement', 'while_statement', 'for_statement', 'try_statement']:
                    complexity += 1
                elif child.type == 'binary_expression':
                    complexity += len(child.children) - 1
        return complexity

def get_tree_sitter_analyzer() -> TreeSitterAnalyzer:
    """Tree-sitter 분석기 인스턴스 반환"""
    return TreeSitterAnalyzer() 