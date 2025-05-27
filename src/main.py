import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
from src.api.routes import router

# 라우터 임포트
# from src.api.openai_compatible import router as openai_router
# from src.api.agent import router as agent_router
# from src.api.project import router as project_router

# load_dotenv()

app = FastAPI(
    title="Open CodeAI",
    description="Continue 플러그인 호환 오픈소스 AI 코드 어시스턴트",
    version="1.0.0",
)

# CORS 허용 (로컬 개발용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
# app.include_router(openai_router, prefix="/v1")
# app.include_router(agent_router, prefix="/api/agent")
# app.include_router(project_router, prefix="/api/project")
app.include_router(router)

@app.get("/")
def root() -> dict:
    return {"message": "Open CodeAI API 서버가 실행 중입니다."}

def main() -> None:
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main() 