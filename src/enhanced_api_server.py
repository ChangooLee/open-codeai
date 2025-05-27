from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
from typing import List, Optional

from src.ai_agent import CodeAIAgent
from src.enhanced_rag_system import EnhancedRAGSystem
from src.file_watcher import AutoIndexingSystem
from src.llm_server import QwenCodeLLMServer

class AgentRequest(BaseModel):
    message: str
    project_path: Optional[str] = None

class AutonomousTaskRequest(BaseModel):
    task_type: str  # "code_review", "fix_issues", "implement_feature"
    target: Optional[str] = None
    description: Optional[str] = None
    project_path: Optional[str] = None

app = FastAPI(title="Enhanced Cursor AI Clone")
agents = {}
auto_indexers = {}

@app.on_event("startup")
async def startup_event():
    print("Enhanced API 서버 시작됨")

@app.post("/api/agent/chat")
async def agent_chat(request: AgentRequest):
    try:
        project_path = request.project_path or "/default/project"
        if project_path not in agents:
            llm_server = QwenCodeLLMServer()
            rag_system = EnhancedRAGSystem(index_dir=f"./data/index_{hash(project_path)}")
            agent = CodeAIAgent(project_path, llm_server)
            agent.tools.rag_system = rag_system
            agents[project_path] = agent
            auto_indexer = AutoIndexingSystem(project_path, rag_system, agent)
            auto_indexer.start_monitoring()
            auto_indexers[project_path] = auto_indexer
        agent = agents[project_path]
        response = await agent.process_message(request.message)
        return {"response": response, "project_path": project_path, "tools_used": agent.context.tools_used[-5:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent/autonomous")
async def autonomous_task(request: AutonomousTaskRequest, background_tasks: BackgroundTasks):
    try:
        project_path = request.project_path or "/default/project"
        if project_path not in agents:
            raise HTTPException(status_code=404, detail="프로젝트가 초기화되지 않았습니다")
        agent = agents[project_path]
        if request.task_type == "code_review":
            background_tasks.add_task(_run_autonomous_review, agent, request.target)
        elif request.task_type == "fix_issues":
            background_tasks.add_task(_run_fix_issues, agent, request.target)
        elif request.task_type == "implement_feature":
            background_tasks.add_task(_run_implement_feature, agent, request.description)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 작업 타입")
        return {
            "message": f"{request.task_type} 작업이 백그라운드에서 시작되었습니다",
            "task_type": request.task_type,
            "target": request.target
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _run_autonomous_review(agent: CodeAIAgent, target: Optional[str]):
    try:
        result = await agent.autonomous_code_review(target)
        print(f"자율적 코드 리뷰 완료: {result[:200]}...")
    except Exception as e:
        print(f"자율적 코드 리뷰 오류: {e}")

async def _run_fix_issues(agent: CodeAIAgent, target: Optional[str]):
    try:
        result = await agent.fix_code_issues(target or "")
        print(f"이슈 수정 완료: {result[:200]}...")
    except Exception as e:
        print(f"이슈 수정 오류: {e}")

async def _run_implement_feature(agent: CodeAIAgent, description: Optional[str]):
    try:
        result = await agent.implement_feature(description or "")
        print(f"기능 구현 완료: {result[:200]}...")
    except Exception as e:
        print(f"기능 구현 오류: {e}")

@app.get("/api/project/stats/{project_path:path}")
async def get_project_stats(project_path: str):
    try:
        if project_path in auto_indexers:
            auto_indexer = auto_indexers[project_path]
            stats = auto_indexer.rag_system.get_project_statistics()
            stats.update(auto_indexer.change_stats)
            return stats
        else:
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/reindex/{project_path:path}")
async def reindex_project(project_path: str, background_tasks: BackgroundTasks):
    try:
        if project_path in agents:
            rag_system = agents[project_path].tools.rag_system
            background_tasks.add_task(_reindex_project, rag_system, project_path)
            return {"message": "재인덱싱이 백그라운드에서 시작되었습니다"}
        else:
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def _reindex_project(rag_system: EnhancedRAGSystem, project_path: str):
    try:
        rag_system.index_file(project_path, force=True)
        print(f"프로젝트 재인덱싱 완료: {project_path}")
    except Exception as e:
        print(f"재인덱싱 오류: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 