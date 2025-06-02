#!/bin/bash
set -e

# scripts 디렉토리 내 모든 .sh 파일 실행 권한 부여
chmod +x scripts/*.sh 2>/dev/null || true

# Docker 환경에서 오프라인 패키지 다운로드 (Python 3.12 기준, 빌드 도구 포함)
echo "[INFO] Docker 컨테이너에서 오프라인 패키지 다운로드를 시작합니다. (Linux x86_64, Python 3.12, 빌드 도구 포함)"
docker run --rm -v "$PWD":/app -w /app python:3.12-slim bash -c '
  apt-get update && \
  apt-get install -y --no-install-recommends make gcc g++ python3-dev && \
  pip install --upgrade pip && \
  pip download -r requirements.txt -d offline_packages
'
if [ $? -eq 0 ]; then
    echo "[INFO] Docker 환경에서 오프라인 패키지 다운로드 완료"
else
    echo "[ERROR] Docker 환경에서 pip download 실패! requirements.txt 또는 네트워크 상태를 확인하세요."
    exit 1
fi

# codebert-base 모델 다운로드 (HuggingFace Hub, 공개 모델)
if ! [ -d offline_packages/codebert-base ]; then
  echo "[INFO] microsoft/codebert-base 모델 다운로드 (공개 모델, 인증 불필요)"
  python3 -m pip install huggingface_hub --quiet
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/codebert-base', local_dir='offline_packages/codebert-base', local_dir_use_symlinks=False)"
  if [ $? -eq 0 ]; then
      echo "[INFO] codebert-base 모델 다운로드 완료"
  else
      echo "[ERROR] codebert-base 모델 다운로드 실패! 네트워크 상태 또는 huggingface_hub 설치를 확인하세요."
      exit 1
  fi
fi

# lxml 등 XML 파서도 requirements.txt에 포함되어야 함 (MyBatis 매퍼 분석 지원)

echo ""
echo "[INFO] 오프라인 설치 준비가 완료되었습니다."
echo ""
echo "===== 오프라인 설치 가이드 ====="
echo "1. offline_packages/와 offline_packages/docker-images/ 폴더를 오프라인 서버로 복사하세요."
echo "2. 오프라인 서버에서 아래 명령을 순서대로 실행하세요:"
echo ""
echo "   # (선택) 가상환경 생성 및 활성화"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo ""
echo "   # 오프라인 Docker 이미지 로드"
echo "   for tar in offline_packages/docker-images/*.tar; do docker load -i \"$tar\"; done"
echo ""
echo "   # 오프라인 패키지 설치 및 서비스 시작"
echo "   bash scripts/install.sh"
echo ""
echo "3. 설치가 완료되면 scripts/start.sh로 서비스를 시작하세요."
echo "==============================="
echo "[INFO] microsoft/codebert-base 모델이 offline_packages/codebert-base 폴더에 준비되어야 합니다." 