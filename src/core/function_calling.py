"""
Open CodeAI - Function Calling 시스템
LLM이 자동으로 코드베이스 검색 및 분석 함수를 호출할 수 있는 시스템
"""
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union, cast
from dataclasses import dataclass
from enum import Enum
import inspect
import os

from ..utils.logger import get_logger
from .rag_system import get_rag_system
from .code_analyzer import get_code_analyzer

logger = get_logger(__name__)

class FunctionType(Enum):
    """함수 타입"""
    SEARCH = "search"
    ANALYZE = "analyze"
    INDEX = "index"
    UTILITY = "utility"

@dataclass
class FunctionParameter:
    """함수 파라미터 정의"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None

@dataclass
class FunctionDefinition:
    """함수 정의"""
    name: str
    description: str
    parameters: List[FunctionParameter]
    function_type: FunctionType
    handler: Callable
    
    def to_openai_format(self) -> Dict[str, Any]:
        """OpenAI function calling 형식으로 변환"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            
            if param.enum is not None and isinstance(param.enum, list):
                prop["enum"] = cast(Any, param.enum)  # type: ignore
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

class FunctionCallRegistry:
    """함수 호출 레지스트리"""
    
    def __init__(self) -> None:
        self.functions: Dict[str, FunctionDefinition] = {}
        self.rag_system = get_rag_system()
        
        # 기본 함수들 등록
        self._register_default_functions()
    
    def register_function(self, func_def: FunctionDefinition) -> None:
        """함수 등록"""
        self.functions[func_def.name] = func_def
        print(f"[FunctionCallRegistry] 함수 등록됨: {func_def.name}")
    
    def _register_default_functions(self) -> None:
        """기본 함수들 등록"""
        
        # 코드베이스 검색 함수
        search_func = FunctionDefinition(
            name="search_codebase",
            description="프로젝트 코드베이스에서 관련 코드를 검색합니다. 함수, 클래스, 변수 등을 찾을 때 사용하세요.",
            parameters=[
                FunctionParameter(
                    name="query",
                    type="string", 
                    description="검색할 내용 (함수명, 클래스명, 기능 설명 등)"
                ),
                FunctionParameter(
                    name="max_results",
                    type="integer",
                    description="최대 결과 수",
                    required=False,
                    default=5
                ),
                FunctionParameter(
                    name="file_type",
                    type="string",
                    description="특정 파일 타입으로 검색 제한",
                    required=False,
                    enum=["python", "javascript", "typescript", "java", "cpp", "go", "rust"]
                )
            ],
            function_type=FunctionType.SEARCH,
            handler=self._search_codebase
        )
        self.register_function(search_func)
        
        # 파일 분석 함수
        analyze_func = FunctionDefinition(
            name="analyze_file",
            description="특정 파일의 구조와 내용을 분석합니다. 파일의 함수, 클래스, 의존성 등을 파악할 때 사용하세요.",
            parameters=[
                FunctionParameter(
                    name="file_path",
                    type="string",
                    description="분석할 파일의 경로"
                ),
                FunctionParameter(
                    name="include_content",
                    type="boolean",
                    description="파일 내용 포함 여부",
                    required=False,
                    default=False
                )
            ],
            function_type=FunctionType.ANALYZE,
            handler=self._analyze_file
        )
        self.register_function(analyze_func)
        
        # 관련 파일 찾기 함수
        find_related_func = FunctionDefinition(
            name="find_related_files",
            description="특정 파일과 관련된 다른 파일들을 찾습니다. 의존성, 임포트 관계 등을 기반으로 찾습니다.",
            parameters=[
                FunctionParameter(
                    name="file_path",
                    type="string",
                    description="기준이 되는 파일 경로"
                ),
                FunctionParameter(
                    name="max_depth",
                    type="integer",
                    description="관계 탐색 깊이",
                    required=False,
                    default=2
                )
            ],
            function_type=FunctionType.SEARCH,
            handler=self._find_related_files
        )
        self.register_function(find_related_func)
        
        # 프로젝트 인덱싱 함수
        index_project_func = FunctionDefinition(
            name="index_project", 
            description="프로젝트 전체를 인덱싱하여 검색 가능하게 합니다. 새로운 프로젝트를 분석하기 전에 사용하세요.",
            parameters=[
                FunctionParameter(
                    name="project_path",
                    type="string",
                    description="인덱싱할 프로젝트 경로"
                ),
                FunctionParameter(
                    name="max_files",
                    type="integer",
                    description="최대 처리할 파일 수",
                    required=False,
                    default=1000
                )
            ],
            function_type=FunctionType.INDEX,
            handler=self._index_project
        )
        self.register_function(index_project_func)
        
        # 인덱싱 상태 확인 함수
        index_stats_func = FunctionDefinition(
            name="get_index_stats",
            description="현재 인덱싱된 프로젝트의 통계 정보를 조회합니다.",
            parameters=[],
            function_type=FunctionType.UTILITY,
            handler=self._get_index_stats
        )
        self.register_function(index_stats_func)
        
        # 함수 정의 찾기
        find_function_func = FunctionDefinition(
            name="find_function_definition",
            description="특정 함수의 정의를 찾습니다. 함수가 어디에 정의되어 있는지 알고 싶을 때 사용하세요.",
            parameters=[
                FunctionParameter(
                    name="function_name",
                    type="string",
                    description="찾을 함수명"
                ),
                FunctionParameter(
                    name="include_usage",
                    type="boolean", 
                    description="함수 사용 예시도 포함할지 여부",
                    required=False,
                    default=True
                )
            ],
            function_type=FunctionType.SEARCH,
            handler=self._find_function_definition
        )
        self.register_function(find_function_func)
        
        # 클래스 정의 찾기
        find_class_func = FunctionDefinition(
            name="find_class_definition",
            description="특정 클래스의 정의와 메서드들을 찾습니다.",
            parameters=[
                FunctionParameter(
                    name="class_name",
                    type="string",
                    description="찾을 클래스명"
                ),
                FunctionParameter(
                    name="include_methods",
                    type="boolean",
                    description="클래스 메서드들도 포함할지 여부", 
                    required=False,
                    default=True
                )
            ],
            function_type=FunctionType.SEARCH,
            handler=self._find_class_definition
        )
        self.register_function(find_class_func)
    
    # 함수 핸들러들
    
    async def _search_codebase(self, query: str, max_results: int = 5, file_type: Optional[str] = None) -> Dict[str, Any]:
        """코드베이스 검색"""
        try:
            if file_type:
                query = f"{query} filetype:{file_type}"
            user_query = extract_user_query(query)
            results = await self.rag_system.search_codebase(user_query, k=max_results)
            
            formatted_results = []
            for result in results:
                chunk = result.chunk
                formatted_result = {
                    "file_path": chunk.file_path,
                    "content": chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                    "line_range": f"{chunk.start_line}-{chunk.end_line}",
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "similarity_score": result.similarity_score,
                    "metadata": chunk.metadata or {}
                }
                
                if result.graph_connections:
                    formatted_result["related_files"] = result.graph_connections[:3]
                
                formatted_results.append(formatted_result)
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }
            
        except Exception as e:
            print(f"[FunctionCallRegistry] 코드베이스 검색 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _find_related_files(self, file_path: str, max_depth: int = 2) -> Dict[str, Any]:
        """관련 파일 찾기"""
        try:
            related_files = self.rag_system.graph_db.find_related_files(file_path, max_depth)
            
            return {
                "status": "success",
                "base_file": file_path,
                "related_files": related_files,
                "total_found": len(related_files),
                "max_depth": max_depth
            }
            
        except Exception as e:
            print(f"[FunctionCallRegistry] 관련 파일 검색 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _index_project(self, project_path: str = None, max_files: int = 1000) -> Dict[str, Any]:
        """프로젝트 인덱싱"""
        try:
            print(f"[FunctionCallRegistry] 프로젝트 인덱싱 시작: {project_path}")
            # Use workspace root as default
            if not project_path:
                from src.config import settings
                project_path = getattr(settings, 'PROJECT_ROOT', os.getcwd())
            result = await self.rag_system.indexer.index_directory(project_path, max_files)
            return {
                "status": "success",
                "message": f"프로젝트 인덱싱 완료: {result.get('success_count', 0)}/{result.get('total_files', 0)} 파일",
                **result
            }
        except Exception as e:
            print(f"[FunctionCallRegistry] 프로젝트 인덱싱 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_index_stats(self) -> Dict[str, Any]:
        """인덱싱 통계 조회"""
        try:
            stats = self.rag_system.indexer.get_indexing_stats()
            
            return {
                "status": "success",
                "statistics": stats
            }
            
        except Exception as e:
            print(f"[FunctionCallRegistry] 통계 조회 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _find_function_definition(self, function_name: str, include_usage: bool = True) -> Dict[str, Any]:
        """함수 정의 찾기"""
        try:
            query = f"def {function_name}" if not function_name.startswith("def ") else function_name
            user_query = extract_user_query(query)
            results = await self.rag_system.search_codebase(user_query, k=10)
            
            function_results = []
            usage_examples = []
            
            for result in results:
                chunk = result.chunk
                
                # 함수 정의 찾기
                if chunk.chunk_type == 'function' and chunk.metadata:
                    if chunk.metadata.get('function_name') == function_name:
                        function_results.append({
                            "file_path": chunk.file_path,
                            "definition": chunk.content,
                            "line_range": f"{chunk.start_line}-{chunk.end_line}",
                            "parameters": chunk.metadata.get('parameters', []),
                            "complexity": chunk.metadata.get('complexity', 1),
                            "docstring": chunk.metadata.get('docstring')
                        })
                
                # 함수 사용 예시 찾기
                elif include_usage and function_name in chunk.content:
                    # 함수 호출 패턴 확인
                    call_patterns = [
                        f"{function_name}(",
                        f".{function_name}(",
                        f" {function_name}("
                    ]
                    
                    if any(pattern in chunk.content for pattern in call_patterns):
                        usage_examples.append({
                            "file_path": chunk.file_path,
                            "usage_context": chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                            "line_range": f"{chunk.start_line}-{chunk.end_line}"
                        })
            
            return {
                "status": "success",
                "function_name": function_name,
                "definitions": function_results,
                "usage_examples": usage_examples[:5] if include_usage else [],
                "total_definitions": len(function_results),
                "total_usages": len(usage_examples)
            }
            
        except Exception as e:
            print(f"[FunctionCallRegistry] 함수 정의 검색 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _find_class_definition(self, class_name: str, include_methods: bool = True) -> Dict[str, Any]:
        """클래스 정의 찾기"""
        try:
            query = f"class {class_name}" if not class_name.startswith("class ") else class_name
            user_query = extract_user_query(query)
            results = await self.rag_system.search_codebase(user_query, k=10)
            
            class_results = []
            
            for result in results:
                chunk = result.chunk
                
                if chunk.chunk_type == 'class' and chunk.metadata:
                    if chunk.metadata.get('class_name') == class_name:
                        class_info = {
                            "file_path": chunk.file_path,
                            "definition": chunk.content,
                            "line_range": f"{chunk.start_line}-{chunk.end_line}",
                            "inheritance": chunk.metadata.get('inheritance', []),
                            "docstring": chunk.metadata.get('docstring')
                        }
                        
                        if include_methods:
                            # 메서드 정보 추가
                            methods = chunk.metadata.get('methods', [])
                            if methods:
                                # 각 메서드의 상세 정보 검색
                                method_details = []
                                for method_name in methods[:10]:  # 최대 10개 메서드
                                    method_query = f"{class_name}.{method_name}"
                                    method_results = await self.rag_system.search_codebase(method_query, k=3)
                                    
                                    for method_result in method_results:
                                        if method_result.chunk.chunk_type == 'function':
                                            method_details.append({
                                                "name": method_name,
                                                "content": method_result.chunk.content[:200] + "..." if len(method_result.chunk.content) > 200 else method_result.chunk.content,
                                                "parameters": method_result.chunk.metadata.get('parameters', []) if method_result.chunk.metadata else []
                                            })
                                            break
                                
                                class_info["methods"] = method_details
                        
                        class_results.append(class_info)
            
            return {
                "status": "success",
                "class_name": class_name,
                "definitions": class_results,
                "total_found": len(class_results)
            }
            
        except Exception as e:
            print(f"[FunctionCallRegistry] 클래스 정의 검색 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_file(self, file_path: str, include_content: bool = False) -> Dict[str, Any]:
        """파일 분석 핸들러 (파일명만 입력해도 자동 경로 매핑)"""
        analyzer = get_code_analyzer()
        # 1. 입력 경로가 실제 파일인지 확인
        if not os.path.exists(file_path):
            # 2. vector DB에서 파일명 기반 경로 검색
            candidates = self.rag_system.vector_db.search_file_by_name(file_path)
            if not candidates:
                # 3. graph DB에서도 시도
                candidates = self.rag_system.graph_db.search_file_by_name(file_path)
            if not candidates:
                return {"status": "error", "message": f"파일을 찾을 수 없습니다: {file_path}"}
            # 4. 가장 유사한 경로 선택 (여러 개면 모두 분석)
            file_path = candidates[0]
        # 5. 기존대로 분석 진행
        analysis = await analyzer.analyze_file(file_path)
        if not analysis:
            return {"status": "error", "message": f"분석 실패: {file_path}"}
        result = {"status": "success", "analysis": analysis}
        if include_content:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                result["content"] = f.read()
        return result
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """사용 가능한 함수 목록 (OpenAI 형식)"""
        return [func_def.to_openai_format() for func_def in self.functions.values()]
    
    async def call_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """함수 호출"""
        try:
            if function_name not in self.functions:
                return {"status": "error", "message": f"함수를 찾을 수 없습니다: {function_name}"}
            
            func_def = self.functions[function_name]
            
            # 파라미터 검증 및 기본값 설정
            validated_args = {}
            
            for param in func_def.parameters:
                if param.required and param.name not in arguments:
                    return {"status": "error", "message": f"필수 파라미터 누락: {param.name}"}
                
                if param.name in arguments:
                    validated_args[param.name] = arguments[param.name]
                elif param.default is not None:
                    validated_args[param.name] = param.default
            
            # 함수 실행
            if asyncio.iscoroutinefunction(func_def.handler):
                result = await func_def.handler(**validated_args)
            else:
                result = func_def.handler(**validated_args)
            
            print(f"[FunctionCallRegistry] 함수 실행 완료: {function_name}")
            if isinstance(result, dict):
                return result
            return {"status": "error", "message": "함수 반환값이 dict가 아님", "raw_result": str(result)}
            
        except Exception as e:
            print(f"[FunctionCallRegistry] 함수 실행 실패 {function_name}: {e}")
            return {"status": "error", "message": str(e)}

class FunctionCallParser:
    """함수 호출 파싱 및 처리"""
    
    def __init__(self, registry: FunctionCallRegistry):
        self.registry = registry
    
    def extract_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """텍스트에서 함수 호출 추출"""
        function_calls = []
        
        # JSON 형태의 함수 호출 패턴 찾기
        json_pattern = r'```json\s*(\{[^`]*\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and 'function' in parsed and 'arguments' in parsed:
                    function_calls.append({
                        'name': parsed['function'],
                        'arguments': parsed['arguments']
                    })
            except json.JSONDecodeError:
                continue
        
        # 함수 호출 패턴도 찾기 (예: search_codebase("query"))
        for func_name in self.registry.functions.keys():
            pattern = rf'{func_name}\s*\(\s*([^)]*)\s*\)'
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    # 간단한 파라미터 파싱 (문자열만 지원)
                    args_str = match.strip()
                    if args_str.startswith('"') and args_str.endswith('"'):
                        function_calls.append({
                            'name': func_name,
                            'arguments': {'query': args_str[1:-1]}  # 첫 번째 파라미터를 query로 가정
                        })
                except Exception:
                    continue
        
        return function_calls
    
    async def process_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """함수 호출들 처리"""
        results = []
        
        for call in function_calls:
            result = await self.registry.call_function(call['name'], call['arguments'])
            results.append({
                'function_name': call['name'],
                'arguments': call['arguments'],
                'result': result
            })
        
        return results

def extract_user_query(query):
    """
    query가 dict(messages=...) 또는 str(prompt)일 때, 가장 마지막 user 메시지 content만 추출
    """
    if isinstance(query, dict) and 'messages' in query:
        user_messages = [m['content'] for m in query['messages'] if m.get('role') == 'user']
        return user_messages[-1] if user_messages else ''
    elif isinstance(query, list):
        user_messages = [m['content'] for m in query if m.get('role') == 'user']
        return user_messages[-1] if user_messages else ''
    elif isinstance(query, str):
        import re
        matches = re.findall(r'User: (.*?)(?:\n|$)', query, re.DOTALL)
        return matches[-1].strip() if matches else query.strip()
    return str(query)

class EnhancedLLMManager:
    """Function Calling이 통합된 LLM 관리자"""
    
    def __init__(self, base_llm_manager: Any) -> None:
        self.base_llm = base_llm_manager
        self.function_registry = FunctionCallRegistry()
        self.function_parser = FunctionCallParser(self.function_registry)
        self.rag_system = get_rag_system()
    
    async def generate_response_with_functions(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_function_calling: bool = True
    ) -> str:
        """Function Calling을 포함한 응답 생성"""
        try:
            if self._needs_codebase_context(prompt):
                user_query = extract_user_query(prompt)
                context = await self.rag_system.get_context_for_query(user_query)
                if context:
                    enhanced_prompt = (
                        "사용자 질문에 답변하기 위해 관련 코드를 검색했습니다.\n\n"
                        "## 관련 코드 컨텍스트:\n" + context + "\n\n"
                        "## 사용자 질문:\n" + user_query + "\n\n"
                        "## 답변 작성 지침 (매우 중요):\n"
                        "- 반드시 먼저 간결한 자연어 설명을 한 문단 이내로 작성하세요.\n"
                        "- 그 다음, 실제 코드 변경이 필요한 경우 아래 예시처럼 info string(언어 + 프로젝트 루트 기준 상대경로)을 포함한 코드블록을 반드시 제공하세요.\n"
                        "- 설명과 코드블록은 반드시 분리하세요.\n"
                        "- 여러 파일을 수정할 경우, 각 파일마다 설명+코드블록 쌍을 반복하세요.\n"
                        "- diff, 삭제, 추가 등 모든 변경도 info string을 포함한 코드블록으로 작성하세요.\n"
                        "- 코드블록 외에는 불필요한 요약, 적용 방법, 추가 설명을 넣지 마세요.\n\n"
                        "예시:\n아래처럼 설명 후 코드블록을 제공합니다.\n\n"
                        "서비스 배열에 autogen을 추가합니다.\n"
                        "```javascript data/services.js\n"
                        "// ... existing code ...\n"
                        "{\n    name: \"autogen\",\n    url: \"http://121.141.60.219:3004/\",\n    icon: \"/icons/autogen.png\",\n    description: \"자동 생성 서비스로, 필요한 데이터나 코드를 자동으로 생성합니다.\"\n},\n"
                        "// ... rest of code ...\n"
                        "```\n\n"
                        "또는 diff 예시:\n새 서비스 추가 diff입니다.\n"
                        "```diff data/services.js\n"
                        "+    {\n"
                        "+        name: \"autogen\",\n"
                        "+        url: \"http://121.141.60.219:3004/\",\n"
                        "+        icon: \"/icons/autogen.png\",\n"
                        "+        description: \"자동 생성 서비스로, 필요한 데이터나 코드를 자동으로 생성합니다.\"\n"
                        "+    },\n"
                        "```\n"
                    )
                else:
                    enhanced_prompt = user_query
            else:
                enhanced_prompt = extract_user_query(prompt)
            
            if enable_function_calling and self._should_use_functions(enhanced_prompt):
                functions_info = self._get_functions_info()
                enhanced_prompt = f"""{enhanced_prompt}

## 사용 가능한 함수들:
{functions_info}

필요한 경우 위의 함수들을 호출하여 더 정확한 정보를 제공할 수 있습니다. 함수 호출이 필요하다면 다음 형식으로 표시하세요:
```json
{{"function": "함수명", "arguments": {{"파라미터": "값"}}}}
```"""
            
            response = await self.base_llm.generate_response(
                prompt=enhanced_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if enable_function_calling:
                function_calls = self.function_parser.extract_function_calls(response)
                
                if function_calls:
                    print(f"[EnhancedLLMManager] 함수 호출 감지: {len(function_calls)}개")
                    
                    call_results = await self._process_function_calls(function_calls)
                    
                    final_prompt = f"""이전 응답: {response}

함수 실행 결과:
{self._format_function_results(call_results)}

위의 함수 실행 결과를 바탕으로 사용자 질문에 대한 최종 답변을 제공하세요:
{prompt}"""
                    
                    final_response = await self.base_llm.generate_response(
                        prompt=final_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    return str(final_response)
            
            return str(response)
            
        except Exception as e:
            print(f"[EnhancedLLMManager] Function calling 응답 생성 실패: {e}")
            return str(await self.base_llm.generate_response(prompt, max_tokens, temperature))
    
    def _needs_codebase_context(self, prompt: str) -> bool:
        """코드베이스 컨텍스트가 필요한지 판단"""
        code_keywords = [
            'function', 'class', 'method', 'variable', 'import', 'module',
            '함수', '클래스', '메서드', '변수', '파일', '코드',
            'how does', 'where is', 'find', 'search', 'show me',
            '어떻게', '어디에', '찾아', '검색', '보여줘'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in code_keywords)
    
    def _should_use_functions(self, prompt: str) -> bool:
        """함수 사용이 필요한지 판단"""
        function_keywords = [
            'analyze', 'index', 'search', 'find', 'list', 'show',
            '분석', '인덱싱', '검색', '찾기', '목록', '보기'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in function_keywords)
    
    def _get_functions_info(self) -> str:
        """사용 가능한 함수 정보 문자열 생성"""
        functions_info = []
        
        for func_def in self.function_registry.functions.values():
            params_info = []
            for param in func_def.parameters:
                param_str = f"{param.name}: {param.type}"
                if not param.required:
                    param_str += " (선택적)"
                if param.description:
                    param_str += f" - {param.description}"
                params_info.append(param_str)
            
            func_info = f"- {func_def.name}: {func_def.description}\n"
            if params_info:
                func_info += f"  파라미터: {', '.join(params_info)}\n"
            
            functions_info.append(func_info)
        
        return "\n".join(functions_info)
    
    async def _process_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """함수 호출 처리
        
        Args:
            function_calls: 함수 호출 정보 리스트
            
        Returns:
            함수 실행 결과 리스트
        """
        results = []
        for call in function_calls:
            try:
                # 필수 키 검증
                func_name = call.get('name')
                args = call.get('arguments')
                if not func_name or args is None:
                    logger.warning(f"잘못된 함수 호출 형식: {call}")
                    continue
                    
                # 함수 실행
                result = await self.registry.call_function(func_name, args)
                
                # 결과 검증
                if not isinstance(result, dict):
                    result = {'status': 'error', 'message': f'잘못된 반환 형식: {type(result)}'}
                
                # 필수 키 추가
                result['function_name'] = func_name
                results.append(result)
                
            except Exception as e:
                logger.error(f"함수 호출 실패 ({func_name}): {e}")
                results.append({
                    'function_name': func_name,
                    'status': 'error',
                    'message': f'함수 실행 중 오류 발생: {str(e)}'
                })
        
        return results

    def _format_function_results(self, results: List[Dict[str, Any]]) -> str:
        """함수 실행 결과 포맷팅
        
        Args:
            results: 함수 실행 결과 리스트
            
        Returns:
            포맷팅된 결과 문자열
        """
        formatted = []
        for result in results:
            func_name = result.get('function_name', 'unknown')
            status = result.get('status', 'unknown')
            message = result.get('message', '')
            
            formatted.append(f"Function: {func_name}")
            formatted.append(f"Status: {status}")
            if message:
                formatted.append(f"Message: {message}")
            formatted.append("---")
        
        return "\n".join(formatted)

# 전역 Function Call Registry 인스턴스
_function_registry_instance = None

def get_function_registry() -> FunctionCallRegistry:
    """전역 Function Call Registry 인스턴스 반환"""
    global _function_registry_instance
    
    if _function_registry_instance is None:
        _function_registry_instance = FunctionCallRegistry()
    
    return _function_registry_instance