from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any

router = APIRouter()

class AgentChatRequest(BaseModel):
    message: str
    project_path: Optional[str] = None

class AgentAutonomousRequest(BaseModel):
    task_type: str
    project_path: str
    options: Optional[Any] = None

@router.post("/chat")
def agent_chat(request: AgentChatRequest):
    # 실제론 core.agent.agent_chat 호출
    return {
        "result": f"(모의 응답) '{request.message}'에 대한 AI 에이전트 답변입니다.",
        "project_path": request.project_path
    }

@router.post("/autonomous")
def agent_autonomous(request: AgentAutonomousRequest):
    # 실제론 core.agent.agent_autonomous 호출
    return {
        "result": f"(모의 응답) '{request.task_type}' 작업을 {request.project_path}에서 수행 완료.",
        "options": request.options
    } 