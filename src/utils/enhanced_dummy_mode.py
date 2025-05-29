"""
Open CodeAI - 향상된 더미 모드 시스템
실제 AI 모델 없이도 현실적이고 유용한 응답을 제공
"""
import os
import re
import json
import random
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from datetime import datetime

class EnhancedDummyLLM:
    """향상된 더미 LLM - 컨텍스트 인식 응답"""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.code_templates = self._load_code_templates()
        self.project_context = self._analyze_project_context()
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """지식 베이스 구축"""
        return {
            'programming_concepts': {
                'python': {
                    'basics': ['variables', 'functions', 'classes', 'modules'],
                    'advanced': ['decorators', 'generators', 'context managers', 'metaclasses'],
                    'frameworks': ['fastapi', 'django', 'flask', 'pytorch', 'tensorflow']
                },
                'javascript': {
                    'basics': ['variables', 'functions', 'objects', 'arrays'],
                    'advanced': ['promises', 'async/await', 'closures', 'prototypes'],
                    'frameworks': ['react', 'vue', 'node.js', 'express']
                },
                'typescript': {
                    'basics': ['types', 'interfaces', 'classes', 'generics'],
                    'advanced': ['decorators', 'mixins', 'conditional types'],
                    'frameworks': ['angular', 'nest.js', 'next.js']
                }
            },
            'common_patterns': {
                'design_patterns': ['singleton', 'factory', 'observer', 'strategy'],
                'architectural_patterns': ['mvc', 'mvp', 'mvvm', 'microservices'],
                'coding_patterns': ['dependency injection', 'repository pattern', 'unit of work']
            },
            'best_practices': {
                'code_quality': ['clean code', 'SOLID principles', 'DRY', 'KISS'],
                'testing': ['unit testing', 'integration testing', 'TDD', 'BDD'],
                'security': ['input validation', 'authentication', 'authorization', 'encryption']
            }
        }
    
    def _load_code_templates(self) -> Dict[str, List[str]]:
        """코드 템플릿 로드"""
        return {
            'python_function': [
                '''def {function_name}({params}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {body}
    return {return_value}''',
                
                '''async def {function_name}({params}):
    """비동기 {description}"""
    try:
        {body}
        return {return_value}
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        raise''',
            ],
            
            'python_class': [
                '''class {class_name}:
    """
    {description}
    
    Attributes:
        {attributes}
    """
    
    def __init__(self, {init_params}):
        {init_body}
    
    def {method_name}(self, {method_params}):
        """주요 메서드"""
        {method_body}
        return {method_return}''',
            ],
            
            'javascript_function': [
                '''function {function_name}({params}) {{
    // {description}
    {body}
    return {return_value};
}}''',
                
                '''const {function_name} = ({params}) => {{
    // {description}
    {body}
    return {return_value};
}};''',

                '''async function {function_name}({params}) {{
    // 비동기 {description}
    try {{
        {body}
        return {return_value};
    }} catch (error) {{
        console.error(`Error in {function_name}:`, error);
        throw error;
    }}
}}''',
            ],
            
            'react_component': [
                '''import React, {{ useState, useEffect }} from 'react';

const {component_name} = ({{ {props} }}) => {{
    const [state, setState] = useState({initial_state});
    
    useEffect(() => {{
        // {effect_description}
        {effect_body}
    }}, [{dependencies}]);
    
    const handle{handler_name} = ({handler_params}) => {{
        {handler_body}
    }};
    
    return (
        <div className="{css_class}">
            {jsx_content}
        </div>
    );
}};

export default {component_name};''',
            ],
            
            'api_endpoint': [
                '''@app.{method}("/{endpoint}")
async def {function_name}({params}):
    """
    {description}
    """
    try:
        {validation}
        
        {business_logic}
        
        return {{
            "status": "success",
            "data": {response_data},
            "message": "{success_message}"
        }}
    except Exception as e:
        logger.error(f"API error: {{e}}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )''',
            ]
        }
    
    def _analyze_project_context(self) -> Dict[str, Any]:
        """현재 프로젝트 컨텍스트 분석"""
        context = {
            'project_type': 'unknown',
            'languages': [],
            'frameworks': [],
            'files': [],
            'structure': {}
        }
        
        try:
            # 프로젝트 루트에서 파일들 분석
            if os.path.exists('package.json'):
                context['project_type'] = 'javascript'
                context['languages'].append('javascript')
                
                # package.json 분석
                try:
                    with open('package.json', 'r') as f:
                        package_data = json.load(f)
                        deps = package_data.get('dependencies', {})
                        if 'react' in deps:
                            context['frameworks'].append('react')
                        if 'vue' in deps:
                            context['frameworks'].append('vue')
                        if 'express' in deps:
                            context['frameworks'].append('express')
                except:
                    pass
            
            if os.path.exists('requirements.txt') or os.path.exists('pyproject.toml'):
                context['project_type'] = 'python'
                context['languages'].append('python')
                
                # requirements.txt 분석
                if os.path.exists('requirements.txt'):
                    try:
                        with open('requirements.txt', 'r') as f:
                            requirements = f.read()
                            if 'fastapi' in requirements:
                                context['frameworks'].append('fastapi')
                            if 'django' in requirements:
                                context['frameworks'].append('django')
                            if 'flask' in requirements:
                                context['frameworks'].append('flask')
                            if 'torch' in requirements:
                                context['frameworks'].append('pytorch')
                    except:
                        pass
            
            # 파일 구조 분석
            common_dirs = ['src', 'lib', 'components', 'utils', 'api', 'models']
            for dir_name in common_dirs:
                if os.path.exists(dir_name):
                    context['structure'][dir_name] = True
                    
                    # 디렉토리 내 파일들 샘플링
                    try:
                        files = list(Path(dir_name).glob('**/*.py'))[:10]
                        context['files'].extend([str(f) for f in files])
                    except:
                        pass
        
        except Exception as e:
            print(f"프로젝트 컨텍스트 분석 오류: {e}")
        
        return context
    
    async def generate_smart_response(self, prompt: str) -> str:
        """스마트한 컨텍스트 기반 응답 생성"""
        
        # 프롬프트 분석
        prompt_lower = prompt.lower()
        
        # 코드 생성 요청 감지
        if any(keyword in prompt_lower for keyword in [
            'write', 'create', 'generate', 'implement', 'build',
            '작성', '생성', '만들', '구현'
        ]):
            return await self._generate_code_response(prompt)
        
        # 설명 요청 감지
        elif any(keyword in prompt_lower for keyword in [
            'explain', 'what is', 'how does', 'describe',
            '설명', '무엇', '어떻게', '설명해'
        ]):
            return await self._generate_explanation_response(prompt)
        
        # 문제 해결 요청 감지
        elif any(keyword in prompt_lower for keyword in [
            'fix', 'debug', 'error', 'problem', 'issue',
            '수정', '디버그', '오류', '문제', '해결'
        ]):
            return await self._generate_troubleshooting_response(prompt)
        
        # 검색/찾기 요청 감지
        elif any(keyword in prompt_lower for keyword in [
            'find', 'search', 'locate', 'where',
            '찾', '검색', '어디'
        ]):
            return await self._generate_search_response(prompt)
        
        # 기본 대화형 응답
        else:
            return await self._generate_conversational_response(prompt)
    
    async def _generate_code_response(self, prompt: str) -> str:
        """코드 생성 응답"""
        
        # 언어 감지
        language = self._detect_language_from_prompt(prompt)
        
        # 요청 타입 감지
        request_type = self._detect_code_request_type(prompt)
        
        if request_type == 'function':
            return self._generate_function_code(prompt, language)
        elif request_type == 'class':
            return self._generate_class_code(prompt, language)
        elif request_type == 'api':
            return self._generate_api_code(prompt, language)
        elif request_type == 'component':
            return self._generate_component_code(prompt, language)
        else:
            return self._generate_generic_code(prompt, language)
    
    def _detect_language_from_prompt(self, prompt: str) -> str:
        """프롬프트에서 언어 감지"""
        prompt_lower = prompt.lower()
        
        # 명시적 언어 지정
        if 'python' in prompt_lower:
            return 'python'
        elif any(lang in prompt_lower for lang in ['javascript', 'js', 'node']):
            return 'javascript'
        elif 'typescript' in prompt_lower:
            return 'typescript'
        elif 'react' in prompt_lower:
            return 'react'
        
        # 프로젝트 컨텍스트 기반
        if 'python' in self.project_context['languages']:
            return 'python'
        elif 'javascript' in self.project_context['languages']:
            return 'javascript'
        
        # 기본값
        return 'python'
    
    def _detect_code_request_type(self, prompt: str) -> str:
        """코드 요청 타입 감지"""
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ['function', 'def', '함수']):
            return 'function'
        elif any(keyword in prompt_lower for keyword in ['class', '클래스']):
            return 'class'
        elif any(keyword in prompt_lower for keyword in ['api', 'endpoint', 'route']):
            return 'api'
        elif any(keyword in prompt_lower for keyword in ['component', '컴포넌트']):
            return 'component'
        else:
            return 'generic'
    
    def _generate_function_code(self, prompt: str, language: str) -> str:
        """함수 코드 생성"""
        
        # 함수명 추출 시도
        function_name = self._extract_function_name(prompt) or "example_function"
        
        # 파라미터 추정
        params = self._estimate_parameters(prompt)
        
        # 템플릿 선택 및 채우기
        if language == 'python':
            template = random.choice(self.code_templates['python_function'])
            
            # 비동기 여부 결정
            is_async = any(keyword in prompt.lower() for keyword in ['async', 'await', '비동기'])
            if is_async:
                template = self.code_templates['python_function'][1]  # async template
            
            return self._fill_python_function_template(template, function_name, params, prompt)
        
        elif language == 'javascript':
            template = random.choice(self.code_templates['javascript_function'])
            return self._fill_javascript_function_template(template, function_name, params, prompt)
        
        else:
            return self._generate_generic_function(function_name, params, prompt)
    
    def _generate_class_code(self, prompt: str, language: str) -> str:
        """클래스 코드 생성"""
        
        class_name = self._extract_class_name(prompt) or "ExampleClass"
        
        if language == 'python':
            template = self.code_templates['python_class'][0]
            return self._fill_python_class_template(template, class_name, prompt)
        else:
            return self._generate_generic_class(class_name, prompt)
    
    def _fill_python_function_template(self, template: str, name: str, params: List[str], prompt: str) -> str:
        """Python 함수 템플릿 채우기"""
        
        # 설명 생성
        description = self._generate_function_description(prompt, name)
        
        # 파라미터 문서 생성
        args_doc = "\n        ".join([f"{param}: 파라미터 설명" for param in params])
        
        # 함수 본문 생성
        body = self._generate_function_body(prompt, name, params)
        
        # 반환값 생성
        return_value = self._generate_return_value(prompt, name)
        
        return template.format(
            function_name=name,
            params=", ".join(params) if params else "",
            description=description,
            args_doc=args_doc,
            return_doc="함수 실행 결과",
            body=body,
            return_value=return_value
        )
    
    def _generate_function_description(self, prompt: str, name: str) -> str:
        """함수 설명 생성"""
        if 'calculate' in prompt.lower() or '계산' in prompt:
            return f"{name}을 통한 계산 수행"
        elif 'process' in prompt.lower() or '처리' in prompt:
            return f"데이터 처리를 위한 {name} 구현"
        elif 'validate' in prompt.lower() or '검증' in prompt:
            return f"입력 데이터 검증을 위한 {name}"
        elif 'convert' in prompt.lower() or '변환' in prompt:
            return f"데이터 변환을 위한 {name}"
        else:
            return f"{name} 함수 - 요청에 따른 기능 구현"
    
    def _generate_function_body(self, prompt: str, name: str, params: List[str]) -> str:
        """함수 본문 생성"""
        lines = []
        
        # 입력 검증
        if params:
            lines.append("# 입력 파라미터 검증")
            for param in params:
                lines.append(f"    if not {param}:")
                lines.append(f"        raise ValueError('Invalid {param}')")
            lines.append("")
        
        # 메인 로직
        if 'calculate' in prompt.lower():
            lines.append("    # 계산 로직")
            lines.append("    result = 0")
            for param in params:
                lines.append(f"    result += {param} if isinstance({param}, (int, float)) else 0")
        
        elif 'process' in prompt.lower():
            lines.append("    # 데이터 처리 로직")
            lines.append("    processed_data = []")
            lines.append("    for item in data if 'data' in locals() else []:")
            lines.append("        processed_item = transform_item(item)")
            lines.append("        processed_data.append(processed_item)")
        
        elif 'api' in prompt.lower() or 'request' in prompt.lower():
            lines.append("    # API 호출 로직")
            lines.append("    headers = {'Content-Type': 'application/json'}")
            lines.append("    response = await make_request(url, headers)")
            lines.append("    return response.json()")
        
        else:
            lines.append("    # 메인 비즈니스 로직")
            lines.append("    logger.info(f'Executing {name} with params: {params}')")
            lines.append("    ")
            lines.append("    # TODO: 실제 구현 로직 추가")
            lines.append("    result = perform_operation()")
        
        return "\n".join(lines)
    
    def _generate_return_value(self, prompt: str, name: str) -> str:
        """반환값 생성"""
        if 'list' in prompt.lower() or '목록' in prompt:
            return "processed_data if 'processed_data' in locals() else []"
        elif 'dict' in prompt.lower() or 'object' in prompt.lower():
            return "{'status': 'success', 'result': result}"
        elif 'bool' in prompt.lower():
            return "True"
        elif 'string' in prompt.lower() or 'str' in prompt.lower():
            return "str(result)"
        else:
            return "result if 'result' in locals() else None"
    
    def _extract_function_name(self, prompt: str) -> Optional[str]:
        """프롬프트에서 함수명 추출"""
        
        # "create function named X" 패턴
        match = re.search(r'function\s+(?:named\s+)?([a-zA-Z_][a-zA-Z0-9_]*)', prompt)
        if match:
            return match.group(1)
        
        # "def X" 패턴
        match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', prompt)
        if match:
            return match.group(1)
        
        # 한글 패턴 "X 함수"
        match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*함수', prompt)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_class_name(self, prompt: str) -> Optional[str]:
        """프롬프트에서 클래스명 추출"""
        
        # "class X" 패턴
        match = re.search(r'class\s+([A-Z][a-zA-Z0-9_]*)', prompt)
        if match:
            return match.group(1)
        
        # "X 클래스" 패턴
        match = re.search(r'([A-Z][a-zA-Z0-9_]*)\s*클래스', prompt)
        if match:
            return match.group(1)
        
        return None
    
    def _estimate_parameters(self, prompt: str) -> List[str]:
        """프롬프트에서 파라미터 추정"""
        params = []
        
        # 일반적인 파라미터 이름들
        common_params = ['data', 'value', 'input', 'text', 'number', 'item', 'config']
        
        # 프롬프트에서 언급된 것들 찾기
        for param in common_params:
            if param in prompt.lower():
                params.append(param)
        
        # 타입 힌트가 있는 경우
        if 'string' in prompt.lower() or 'str' in prompt.lower():
            params.append('text: str')
        elif 'number' in prompt.lower() or 'int' in prompt.lower():
            params.append('number: int')
        elif 'list' in prompt.lower():
            params.append('items: List[Any]')
        
        return params[:3] if params else ['data']  # 최대 3개 파라미터
    
    async def _generate_explanation_response(self, prompt: str) -> str:
        """설명 응답 생성"""
        
        # 설명 대상 키워드 추출
        topics = self._extract_explanation_topics(prompt)
        
        if not topics:
            return """안녕하세요! Open CodeAI가 더미 모드로 실행 중입니다.

실제 AI 모델이 로드되면 더 정확하고 상세한 설명을 제공할 수 있습니다.

**일반적인 프로그래밍 개념들:**

🔧 **함수 (Functions)**
- 재사용 가능한 코드 블록
- 입력(파라미터)을 받아 처리 후 결과 반환
- 코드의 모듈화와 유지보수성 향상

📦 **클래스 (Classes)**  
- 객체 지향 프로그래밍의 기본 단위
- 데이터(속성)와 기능(메서드)을 하나로 묶음
- 코드 재사용성과 확장성 제공

🔄 **비동기 프로그래밍 (Async Programming)**
- Non-blocking 코드 실행
- I/O 작업의 효율성 개선
- 동시성(Concurrency) 구현

더 구체적인 질문을 해주시면 해당 주제에 맞는 설명을 제공해드리겠습니다!"""
        
        # 주제별 맞춤 설명 생성
        explanations = []
        
        for topic in topics:
            if topic in self.knowledge_base['programming_concepts']:
                explanations.append(self._generate_concept_explanation(topic))
            elif topic in ['function', 'class', 'variable']:
                explanations.append(self._generate_basic_explanation(topic))
        
        if explanations:
            return "\n\n".join(explanations)
        else:
            return self._generate_generic_explanation(prompt)
    
    def _extract_explanation_topics(self, prompt: str) -> List[str]:
        """설명 요청에서 주제 추출"""
        topics = []
        prompt_lower = prompt.lower()
        
        # 프로그래밍 개념들
        concepts = ['function', 'class', 'variable', 'loop', 'condition', 'async', 'api', 'database']
        
        for concept in concepts:
            if concept in prompt_lower:
                topics.append(concept)
        
        # 언어별 키워드
        if 'python' in prompt_lower:
            topics.append('python')
        elif 'javascript' in prompt_lower:
            topics.append('javascript')
        
        return topics
    
    async def _generate_search_response(self, prompt: str) -> str:
        """검색 요청 응답 (프로젝트 파일 기반)"""
        
        # 검색 대상 추출
        search_terms = self._extract_search_terms(prompt)
        
        # 프로젝트 파일들에서 검색 시뮬레이션
        found_files = self._simulate_file_search(search_terms)
        
        if found_files:
            response = "🔍 **검색 결과** (더미 모드)\n\n"
            response += "실제 모델이 로드되면 정확한 코드 검색이 가능합니다.\n\n"
            response += "**발견된 관련 파일들:**\n\n"
            
            for file_info in found_files:
                response += f"📄 **{file_info['path']}**\n"
                response += f"   - 타입: {file_info['type']}\n"
                response += f"   - 크기: {file_info['size']} bytes\n"
                if file_info.get('preview'):
                    response += f"   - 미리보기: {file_info['preview']}\n"
                response += "\n"
            
            response += "**다음 단계:**\n"
            response += "1. 실제 모델 다운로드: `python scripts/download_models.py`\n"
            response += "2. 프로젝트 인덱싱: `python scripts/index_project.py /path/to/project`\n"
            response += "3. 정확한 코드 검색 사용\n"
            
            return response
        else:
            return f"""🔍 **'{' '.join(search_terms)}' 검색 결과**

더미 모드에서는 제한적인 검색만 가능합니다.

**실제 기능 활성화 방법:**
1. AI 모델 다운로드
2. 프로젝트 인덱싱
3. RAG 시스템 활성화

그러면 다음과 같은 고급 검색이 가능합니다:
- 함수 정의 위치 찾기
- 클래스 사용 예시 검색  
- 의존성 관계 분석
- 코드 패턴 매칭"""
    
    def _extract_search_terms(self, prompt: str) -> List[str]:
        """검색어 추출"""
        # 간단한 키워드 추출
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', prompt)
        
        # 불용어 제거
        stop_words = {'find', 'search', 'where', 'is', 'the', 'in', 'for', 'how', 'what'}
        return [word for word in words if word.lower() not in stop_words][:5]
    
    def _simulate_file_search(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """파일 검색 시뮬레이션"""
        found_files = []
        
        # 실제 프로젝트 파일들 스캔
        try:
            for root, dirs, files in os.walk('.'):
                # 불필요한 디렉토리 제외
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                        file_path = os.path.join(root, file)
                        
                        # 검색어와 매치 확인
                        if any(term.lower() in file.lower() for term in search_terms):
                            try:
                                stat = os.stat(file_path)
                                file_info = {
                                    'path': file_path,
                                    'type': file.split('.')[-1],
                                    'size': stat.st_size,
                                    'preview': self._get_file_preview(file_path)
                                }
                                found_files.append(file_info)
                                
                                if len(found_files) >= 5:  # 최대 5개 파일
                                    break
                            except:
                                continue
                
                if len(found_files) >= 5:
                    break
        except:
            pass
        
        return found_files
    
    def _get_file_preview(self, file_path: str) -> str:
        """파일 미리보기 생성"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(200)  # 첫 200자
                return content.strip().replace('\n', ' ')[:100] + '...'
        except:
            return "미리보기 불가"
    
    async def _generate_troubleshooting_response(self, prompt: str) -> str:
        """문제 해결 응답 생성"""
        
        # 오류 유형 감지
        error_type = self._detect_error_type(prompt)
        
        base_response = "🔧 **문제 해결 도움** (더미 모드)\n\n"
        
        if error_type == 'import_error':
            return base_response + """**Import Error 해결 방법:**

1. **패키지 설치 확인**
   ```bash
   pip install 패키지명
   pip list | grep 패키지명
   ```

2. **가상환경 확인**
   ```bash
   which python
   pip --version
   ```

3. **Python 경로 확인**
   ```python
   import sys
   print(sys.path)
   ```

**일반적인 해결책:**
- 가상환경 재활성화: `source venv/bin/activate`
- 패키지 재설치: `pip uninstall && pip install`
- 캐시 정리: `pip cache purge`"""

        elif error_type == 'syntax_error':
            return base_response + """**Syntax Error 해결 방법:**

1. **문법 검사**
   - 괄호, 대괄호, 중괄호 짝 맞추기
   - 들여쓰기 일관성 확인
   - 콜론(:) 누락 확인

2. **디버깅 도구 사용**
   ```bash
   python -m py_compile your_file.py
   flake8 your_file.py
   ```

3. **IDE 도움 받기**
   - VS Code Python 확장
   - PyCharm 문법 검사
   - 실시간 오류 표시 활용"""

        elif error_type == 'api_error':
            return base_response + """**API Error 해결 방법:**

1. **연결 확인**
   ```bash
   curl -I http://localhost:8000/health
   ping localhost
   ```

2. **서버 상태 확인**
   ```bash
   ./start.sh status
   docker ps
   ```

3. **로그 확인**
   ```bash
   tail -f logs/opencodeai.log
   ```

**일반적인 해결책:**
- 서버 재시작: `./start.sh restart`
- 포트 확인: `netstat -tulpn | grep 8000`
- 방화벽 설정 확인"""

        else:
            return base_response + """**일반적인 문제 해결 순서:**

1. **오류 메시지 분석**
   - 정확한 오류 내용 파악
   - 스택 트레이스 확인
   - 오류 발생 위치 추적

2. **기본 확인사항**
   - Python 버전 호환성
   - 의존성 패키지 설치
   - 환경 변수 설정

3. **로그 및 디버깅**
   - 상세 로그 활성화
   - 단계별 디버깅
   - 간단한 테스트 케이스 작성

4. **도움 받기**
   - 공식 문서 확인
   - 커뮤니티 포럼 검색
   - GitHub Issues 확인

**Open CodeAI 관련 문제:**
- 시스템 검증: `python scripts/system_check.py`
- 설치 재실행: `./scripts/install.sh`
- 모델 다운로드: `python scripts/download_models.py`"""
    
    def _detect_error_type(self, prompt: str) -> str:
        """오류 유형 감지"""
        prompt_lower = prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ['import', 'module', 'no module']):
            return 'import_error'
        elif any(keyword in prompt_lower for keyword in ['syntax', 'invalid syntax']):
            return 'syntax_error'
        elif any(keyword in prompt_lower for keyword in ['api', 'connection', 'server', 'request']):
            return 'api_error'
        elif any(keyword in prompt_lower for keyword in ['install', 'setup', 'configuration']):
            return 'setup_error'
        else:
            return 'general'
    
    async def _generate_conversational_response(self, prompt: str) -> str:
        """일반 대화형 응답"""
        
        prompt_lower = prompt.lower()
        
        # 인사말
        if any(greeting in prompt_lower for greeting in ['hello', 'hi', '안녕', '헬로']):
            return """안녕하세요! 👋 Open CodeAI입니다.

현재 **더미 모드**로 실행 중입니다. 기본적인 도움은 드릴 수 있지만, 실제 AI 모델을 로드하면 훨씬 더 정확하고 유용한 도움을 받을 수 있습니다.

**가능한 도움:**
- 🔍 코드 검색 및 분석
- 💡 프로그래밍 질문 답변  
- 🐛 문제 해결 가이드
- 📚 기술 문서 설명

**실제 모델 활성화:**
```bash
python scripts/download_models.py
./start.sh
```

무엇을 도와드릴까요?"""

        # 상태 문의
        elif any(keyword in prompt_lower for keyword in ['status', 'how are you', '상태', '어때']):
            return f"""🤖 **Open CodeAI 상태 보고**

**현재 모드:** 더미 모드 (Dummy Mode)
**실행 시간:** {self._get_uptime()}
**시스템:** 정상 작동 중

**활성화된 기능:**
✅ 기본 대화 시스템
✅ 코드 템플릿 생성
✅ 문제 해결 가이드
✅ 프로젝트 파일 스캔

**비활성화된 기능:**
❌ 실제 AI 추론
❌ 정확한 코드 분석
❌ RAG 시스템
❌ Function Calling

**모든 기능을 활성화하려면:**
1. `python scripts/download_models.py` - 모델 다운로드
2. `./start.sh` - 서버 시작
3. Continue.dev 연결"""

        # 기능 문의
        elif any(keyword in prompt_lower for keyword in ['what can you do', '무엇', '기능']):
            return """🚀 **Open CodeAI 기능 소개**

**현재 가능한 기능 (더미 모드):**

📝 **코드 생성**
- Python/JavaScript 함수 생성
- 클래스 및 컴포넌트 템플릿
- API 엔드포인트 코드

🔍 **프로젝트 분석**
- 파일 구조 탐색
- 기본적인 코드 검색
- 의존성 정보 표시

💡 **학습 도움**
- 프로그래밍 개념 설명
- 코딩 패턴 안내
- 문제 해결 가이드

**실제 모델 로드 시 추가 기능:**
- 🧠 정확한 코드 분석 및 생성
- 🔗 프로젝트 전체 맥락 이해
- ⚡ 실시간 함수 호출
- 🎯 맞춤형 코드 리뷰

**사용 예시:**
- "Python 데이터 처리 함수 만들어줘"
- "이 오류 어떻게 해결하지?"
- "React 컴포넌트 구조 설명해줘" """

        # 도움 요청
        elif any(keyword in prompt_lower for keyword in ['help', 'guide', '도움', '가이드']):
            return """📖 **Open CodeAI 사용 가이드**

**1. 코드 생성 요청하기**
```
"Python에서 JSON 파일 읽는 함수 만들어줘"
"React 로그인 컴포넌트 생성해"
"FastAPI 회원가입 엔드포인트 작성"
```

**2. 문제 해결 요청하기**
```
"ImportError: No module named 'xxx' 해결 방법"
"서버가 안 켜져요"
"코드 실행 시 오류 발생"
```

**3. 설명 요청하기**
```
"비동기 프로그래밍이 뭐야?"
"이 코드 어떻게 작동하는지 설명해줘"
"REST API가 뭔가요?"
```

**4. 검색 요청하기**
```
"user_login 함수 어디 있어?"
"데이터베이스 관련 코드 찾아줘"
"API 호출하는 부분 보여줘"
```

**팁:** 구체적으로 질문할수록 더 좋은 답변을 받을 수 있습니다!"""

        else:
            return f"""Open CodeAI가 더미 모드로 응답드립니다.

현재 입력하신 내용: "{prompt[:100]}{'...' if len(prompt) > 100 else ''}"

더 구체적인 질문을 해주시면 도움을 드릴 수 있습니다:

**예시 질문들:**
- "Python 함수 만들어줘"
- "이 오류 해결 방법 알려줘"  
- "React 컴포넌트 구조 설명해줘"
- "서버 상태 확인하고 싶어"

실제 AI 모델을 로드하면 더 정확하고 맞춤형 답변을 제공할 수 있습니다."""
    
    def _get_uptime(self) -> str:
        """가동 시간 반환 (시뮬레이션)"""
        return f"{random.randint(1, 48)}분 {random.randint(10, 59)}초"
    
    async def generate_smart_embedding(self, text: str) -> List[float]:
        """스마트한 임베딩 생성 (해시 기반이지만 더 현실적)"""
        
        # 텍스트 정규화
        normalized_text = text.lower().strip()
        
        # 여러 해시 조합으로 더 현실적인 임베딩 생성
        embeddings = []
        
        # SHA-256 기반
        hash1 = hashlib.sha256(normalized_text.encode()).hexdigest()
        for i in range(0, len(hash1), 2):
            val = int(hash1[i:i+2], 16) / 255.0 * 2 - 1
            embeddings.append(val)
        
        # MD5 기반 (다른 패턴)
        hash2 = hashlib.md5(normalized_text.encode()).hexdigest()
        for i in range(0, len(hash2), 2):
            val = int(hash2[i:i+2], 16) / 255.0 * 2 - 1
            embeddings.append(val)
        
        # 텍스트 길이와 특성 반영
        text_features = [
            len(text) / 1000.0,  # 텍스트 길이
            text.count(' ') / 100.0,  # 단어 수
            text.count('\n') / 50.0,  # 줄 수
            len(set(text.lower())) / 100.0,  # 고유 문자 수
        ]
        embeddings.extend(text_features)
        
        # 1024 차원으로 맞추기
        while len(embeddings) < 1024:
            # 기존 값들의 조합으로 확장
            embeddings.extend(embeddings[:min(16, 1024 - len(embeddings))])
        
        # 정규화
        import math
        norm = math.sqrt(sum(x*x for x in embeddings[:1024]))
        if norm > 0:
            embeddings = [x/norm for x in embeddings[:1024]]
        
        return embeddings[:1024]


class SmartDummyRAG:
    """스마트한 더미 RAG 시스템"""
    
    def __init__(self):
        self.dummy_llm = EnhancedDummyLLM()
        self.file_cache = {}
        self.simple_index = {}
    
    async def search_and_respond(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """검색 및 응답 생성"""
        
        # 간단한 파일 인덱스 구축 (실행 시점)
        if not self.simple_index:
            await self._build_simple_index()
        
        # 키워드 기반 검색
        results = self._keyword_search(query, max_results)
        
        # 컨텍스트 기반 응답 생성
        response = await self._generate_contextual_response(query, results)
        
        return {
            'query': query,
            'results': results,
            'response': response,
            'total_found': len(results)
        }
    
    async def _build_simple_index(self):
        """간단한 파일 인덱스 구축"""
        
        try:
            for root, dirs, files in os.walk('.'):
                # 무시할 디렉토리
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                          d not in ['__pycache__', 'node_modules', 'venv']]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.md')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()[:2000]  # 첫 2000자만
                            
                            # 키워드 추출
                            keywords = self._extract_keywords(content, file_path)
                            
                            self.simple_index[file_path] = {
                                'content': content,
                                'keywords': keywords,
                                'type': file.split('.')[-1],
                                'size': len(content)
                            }
                            
                        except Exception:
                            continue
        except Exception as e:
            print(f"인덱스 구축 오류: {e}")
    
    def _extract_keywords(self, content: str, file_path: str) -> List[str]:
        """키워드 추출"""
        keywords = []
        
        # 파일명에서 키워드
        filename = os.path.basename(file_path).lower()
        keywords.extend(filename.replace('.', ' ').split())
        
        # 함수명 추출
        function_matches = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        keywords.extend(function_matches)
        
        # 클래스명 추출
        class_matches = re.findall(r'class\s+([A-Z][a-zA-Z0-9_]*)', content)
        keywords.extend(class_matches)
        
        # import 추출
        import_matches = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)', content)
        keywords.extend(import_matches)
        
        # 변수명 추출 (간단한 패턴)
        var_matches = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', content)
        keywords.extend(var_matches[:10])  # 최대 10개
        
        return list(set(keywords))  # 중복 제거
    
    def _keyword_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """키워드 기반 검색"""
        
        query_words = query.lower().split()
        results = []
        
        for file_path, file_data in self.simple_index.items():
            score = 0
            content_lower = file_data['content'].lower()
            keywords_lower = [k.lower() for k in file_data['keywords']]
            
            # 키워드 매칭 점수
            for word in query_words:
                # 파일 내용에서 검색
                score += content_lower.count(word) * 2
                
                # 키워드에서 검색
                for keyword in keywords_lower:
                    if word in keyword:
                        score += 5
                
                # 파일명에서 검색
                if word in file_path.lower():
                    score += 10
            
            if score > 0:
                results.append({
                    'file_path': file_path,
                    'score': score,
                    'type': file_data['type'],
                    'preview': file_data['content'][:200] + '...',
                    'keywords': file_data['keywords'][:5]
                })
        
        # 점수순 정렬
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:max_results]
    
    async def _generate_contextual_response(self, query: str, results: List[Dict]) -> str:
        """컨텍스트 기반 응답 생성"""
        
        if not results:
            return await self.dummy_llm.generate_smart_response(query)
        
        # 검색 결과를 포함한 응답 생성
        context = f"검색 결과: {len(results)}개의 관련 파일을 찾았습니다.\n\n"
        
        for i, result in enumerate(results, 1):
            context += f"{i}. **{result['file_path']}** (점수: {result['score']})\n"
            context += f"   타입: {result['type']}\n"
            context += f"   키워드: {', '.join(result['keywords'])}\n"
            context += f"   미리보기: {result['preview'][:100]}...\n\n"
        
        enhanced_query = f"다음 검색 결과를 바탕으로 질문에 답변해주세요:\n\n{context}\n질문: {query}"
        
        response = await self.dummy_llm.generate_smart_response(enhanced_query)
        
        return f"{context}\n**AI 응답:**\n{response}"


# 전역 인스턴스들
_enhanced_dummy_llm = None
_smart_dummy_rag = None

def get_enhanced_dummy_llm() -> EnhancedDummyLLM:
    """향상된 더미 LLM 인스턴스 반환"""
    global _enhanced_dummy_llm
    
    if _enhanced_dummy_llm is None:
        _enhanced_dummy_llm = EnhancedDummyLLM()
    
    return _enhanced_dummy_llm

def get_smart_dummy_rag() -> SmartDummyRAG:
    """스마트 더미 RAG 인스턴스 반환"""
    global _smart_dummy_rag
    
    if _smart_dummy_rag is None:
        _smart_dummy_rag = SmartDummyRAG()
    
    return _smart_dummy_rag