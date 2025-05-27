from fastapi import APIRouter
from typing import Dict

router = APIRouter()

# 엔드포인트 예시
@router.get("/ping")
def ping() -> Dict:
    return {"message": "pong"} 