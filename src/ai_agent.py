import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.agent_tools import CodebaseTools, TOOL_SCHEMAS, ToolResult

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class AgentContext:
    """에이전트 컨텍스트"""
    project_path: str
    conversation_history: List[ChatMessage]
    current_task: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)

class CodeAIAgent:
    """Function Calling 기반 AI 코드 에이전트"""
    
    def __init__(self, project_path: str, llm_server):
        self.project_path = project_path
        self.llm = llm_server
        self.tools = CodebaseTools(project_path)
        self.context = AgentContext(
            project_path=project_path,
            conversation_history=[],
            tools_used=[]
        )
        self.tool_functions = {
            "read_file": self.tools.read_file,
            "write_file": self.tools.write_file,
            "list_files": self.tools.list_files,
            "analyze_code_structure": self.tools.analyze_code_structure,
            "search_code": self.tools.search_code,
            "run_tests": self.tools.run_tests,
            "lint_code": self.tools.lint_code,
            "get_git_diff": self.tools.get_git_diff
        }

    def _create_system_prompt(self) -> str:
        return f"""You are an advanced AI coding assistant working on project: {self.project_path}

Your capabilities include:
- Reading and writing files
- Analyzing code structure 
- Searching through codebase
- Running tests and linting
- Monitoring git changes
- Autonomous code review and improvements

You can call functions to gather information and take actions. Always:
1. Understand the user's request thoroughly
2. Plan your approach by breaking down complex tasks
3. Use appropriate tools to gather necessary information
4. Provide detailed explanations of your analysis
5. Suggest concrete improvements with code examples
6. Verify your changes by running tests when applicable

Current project context:
- Project path: {self.project_path}
- Available tools: {', '.join(self.tool_functions.keys())}

Be proactive and thorough in your analysis."""

    async def process_message(self, user_message: str) -> str:
        self.context.conversation_history.append(
            ChatMessage(role="user", content=user_message)
        )
        messages = [
            ChatMessage(role="system", content=self._create_system_prompt()),
            *self.context.conversation_history
        ]
        max_iterations = 10
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            response = await self._generate_with_tools(messages)
            if response.get("tool_calls"):
                tool_results = await self._execute_tool_calls(response["tool_calls"])
                self.context.conversation_history.append(
                    ChatMessage(role="assistant", content=response.get("content", ""))
                )
                for tool_result in tool_results:
                    self.context.conversation_history.append(
                        ChatMessage(
                            role="user", 
                            content=f"Tool result: {json.dumps(tool_result, indent=2)}"
                        )
                    )
                messages = [
                    ChatMessage(role="system", content=self._create_system_prompt()),
                    *self.context.conversation_history
                ]
            else:
                final_response = response.get("content", "")
                self.context.conversation_history.append(
                    ChatMessage(role="assistant", content=final_response)
                )
                return final_response
        return "작업이 너무 복잡합니다. 단계적으로 나누어 다시 요청해 주세요."

    async def _generate_with_tools(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        tools_description = self._format_tools_for_qwen()
        enhanced_messages = messages.copy()
        enhanced_messages[0].content += f"\n\nAvailable tools:\n{tools_description}"
        enhanced_messages[0].content += "\n\nTo call a tool, respond with JSON format: {\"tool_calls\": [{\"name\": \"tool_name\", \"arguments\": {...}}]}"
        response_text = await self.llm.generate_response(
            enhanced_messages,
            max_tokens=4096,
            temperature=0.1
        )
        try:
            if response_text.strip().startswith("{") and "tool_calls" in response_text:
                parsed_response = json.loads(response_text)
                return parsed_response
            else:
                return {"content": response_text}
        except json.JSONDecodeError:
            return {"content": response_text}

    def _format_tools_for_qwen(self) -> str:
        tools_info = []
        for schema in TOOL_SCHEMAS:
            func_info = schema["function"]
            tools_info.append(f"""
Function: {func_info['name']}
Description: {func_info['description']}
Parameters: {json.dumps(func_info['parameters'], indent=2)}
""")
        return "\n".join(tools_info)

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            if tool_name in self.tool_functions:
                try:
                    result = self.tool_functions[tool_name](**arguments)
                    self.context.tools_used.append(tool_name)
                    results.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": result.__dict__
                    })
                except Exception as e:
                    results.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": {"success": False, "error": str(e)}
                    })
            else:
                results.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": {"success": False, "error": f"Unknown tool: {tool_name}"}
                })
        return results

    async def autonomous_code_review(self, target_path: Optional[str] = None) -> str:
        if target_path:
            review_query = f"Please perform a comprehensive code review of {target_path}. Analyze the code structure, identify potential issues, suggest improvements, and run relevant tests."
        else:
            review_query = "Please perform a comprehensive code review of the entire project. Start by exploring the project structure, then analyze key files, identify issues, and suggest improvements."
        return await self.process_message(review_query)

    async def fix_code_issues(self, file_path: str) -> str:
        fix_query = f"Please analyze {file_path}, identify any code issues (bugs, style problems, performance issues), and fix them. Run linting and tests to verify the fixes."
        return await self.process_message(fix_query)

    async def implement_feature(self, description: str) -> str:
        feature_query = f"Please implement the following feature: {description}. Analyze the existing codebase to understand the architecture, implement the feature following the established patterns, add tests, and ensure everything works correctly."
        return await self.process_message(feature_query) 