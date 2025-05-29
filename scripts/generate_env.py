"""
Open CodeAI .env 자동 생성 스크립트
-----------------------------------
- config.yaml → .env 환경변수 파일로 변환
- 주요 옵션(모델/DB/포트/Neo4j/Redis 등) 자동 추출
- 오프라인/컨테이너리스/샤딩 등 다양한 모드 지원

[사용법/Usage]
$ python scripts/generate_env.py
  (기본: ./configs/config.yaml → ./.env)
$ CONFIG_PATH=./my_config.yaml ENV_PATH=./my.env python scripts/generate_env.py

[주요 옵션/Key Options]
- llm.main_model.path: LLM 모델 경로
- database.graph.type: "neo4j" or "networkx" (컨테이너리스)
- database.vector.sharding: true/false (샤딩)
- performance.gpu.mixed_precision: true/false
- continue_integration.enabled: true/false

[실행 예시/Example]
- 오프라인/컨테이너리스: config.yaml에서 graph.type: networkx 지정
- 샤딩: vector.sharding: true
- 최소 모드: continue_integration.enabled: false, monitoring.metrics.enable: false
"""
import yaml  # type: ignore
import os

CONFIG_PATH = os.environ.get("CONFIG_PATH", "./configs/config.yaml")
ENV_PATH = os.environ.get("ENV_PATH", ".env")

def main() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_lines = []
    # 프로젝트/서버
    env_lines.append(f"PROJECT_NAME={config['project']['name']}")
    env_lines.append(f"VERSION={config['project']['version']}")
    env_lines.append(f"HOST={config['server']['host']}")
    env_lines.append(f"PORT={config['server']['port']}")
    env_lines.append(f"LOG_LEVEL={config['server']['log_level']}")
    # 모델 경로
    env_lines.append(f"MODEL_PATH={config['llm']['main_model']['path']}")
    env_lines.append(f"EMBEDDING_MODEL_PATH={config['llm']['embedding_model']['path']}")
    # DB 경로/타입
    env_lines.append(f"VECTOR_INDEX_PATH={config['database']['vector']['path']}")
    env_lines.append(f"GRAPH_DB_PATH={config['database']['graph']['path']}")
    env_lines.append(f"GRAPH_DB_TYPE={config['database']['graph']['type']}")
    env_lines.append(f"METADATA_DB_PATH={config['database']['metadata']['path']}")
    # Neo4j/Redis
    if config['database']['graph']['type'] == 'neo4j':
        env_lines.append("NEO4J_URI=bolt://localhost:7687")
        env_lines.append("NEO4J_USER=neo4j")
        env_lines.append("NEO4J_PASSWORD=opencodeai")
    if 'redis' in config['database']:
        env_lines.append(f"REDIS_URL={config['database']['redis']['url']}")
    # Continue.dev
    if config.get('continue_integration', {}).get('enabled', False):
        env_lines.append(f"CONTINUE_PORT={config['continue_integration']['port']}")
        env_lines.append(f"CONTINUE_AUTH_TOKEN={config['continue_integration']['auth_token']}")
    # GPU/성능
    env_lines.append(f"GPU_MEMORY_FRACTION={config['llm']['main_model'].get('gpu_memory_fraction', 0.7)}")
    env_lines.append(f"USE_MIXED_PRECISION={str(config['performance']['gpu'].get('mixed_precision', True)).lower()}")
    # 프로젝트 경로
    env_lines.append(f"PROJECT_PATH={config['project'].get('project_path', './')}")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("# Open CodeAI 자동 생성 환경 변수 파일\n")
        for line in env_lines:
            f.write(line + "\n")
    print(f"[generate_env.py] .env 파일 생성 완료: {ENV_PATH}")

if __name__ == "__main__":
    main() 