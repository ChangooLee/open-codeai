#!/bin/bash
set -u  # -e 제거, -u(정의 안된 변수 사용시 에러)

failed_downloads=()

# scripts 디렉토리 내 모든 .sh 파일 실행 권한 부여
chmod +x scripts/*.sh 2>/dev/null || true

# tree-sitter-languages wheel 빌드/복사 기능 제거, PyPI wheel 다운로드 안내만 출력

echo "[INFO] tree-sitter-languages는 PyPI에 wheel이 제공됩니다. 오프라인 설치 시 아래 명령으로 최신 버전(1.10.2) wheel을 offline_packages에 다운로드하세요:"
echo "      pip download tree-sitter-languages==1.10.2 --only-binary=:all: -d offline_packages/"
echo "[TIP] requirements.txt에 tree-sitter-languages==1.10.2를 명시하세요."

# Docker 환경에서 오프라인 패키지 다운로드 (Python 3.12 기준, 빌드 도구 포함)
echo "[INFO] Docker 컨테이너에서 오프라인 패키지 다운로드를 시작합니다. (Linux x86_64, Python 3.12, 빌드 도구 포함)"
docker run --rm -v "$PWD":/app -w /app python:3.12-slim bash -c '
  apt-get update && \
  apt-get install -y --no-install-recommends make gcc g++ python3-dev && \
  pip install --upgrade pip && \
  pip download -r requirements.txt -d offline_packages && \
  pip download torch torchvision torchaudio -d offline_packages
'
if [ $? -eq 0 ]; then
    echo "[INFO] Docker 환경에서 오프라인 패키지 다운로드 완료"
else
    echo "[ERROR] Docker 환경에서 pip download 실패! requirements.txt 또는 네트워크 상태를 확인하세요."
    failed_downloads+=("docker_pip_download")
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
      failed_downloads+=("codebert-base")
  fi
fi

# lxml 등 XML 파서도 requirements.txt에 포함되어야 함 (MyBatis 매퍼 분석 지원)

# tree-sitter-<lang> 패키지는 requirements.txt에서 제거됨. tree-sitter-languages wheel만 offline_packages에 복사하세요.
# 오프라인 환경에서는 아래 명령으로 설치:
#   pip install --no-index --find-links=offline_packages tree_sitter_languages-*.whl

# requirements.txt에 tree-sitter, tree-sitter-languages, lxml, toml, ruamel.yaml, python-frontmatter, markdown, configparser 등 다양한 파서/분석 패키지가 반드시 포함되어야 함

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

echo ""
if [ ${#failed_downloads[@]} -ne 0 ]; then
    echo "==== 다운로드 실패 목록 ===="
    for pkg in "${failed_downloads[@]}"; do
        echo "  - $pkg"
    done
    exit 1
else
    echo "✅ 모든 패키지/이미지/모델 다운로드 성공"
fi

# =====================
# [NEW] 모든 requirements.txt 패키지에 대해 주요 플랫폼/파이썬 버전별 whl 다운로드
# =====================

PKGS=$(grep -vE '^#|^$' requirements.txt | awk -F '==' '{print $1 "==" $2}')
PLATFORMS=("win_amd64" "macosx_11_0_arm64" "manylinux_2_17_x86_64")
PYTHONS=("39" "310" "311" "312")

for pkg in $PKGS; do
  for plat in "${PLATFORMS[@]}"; do
    for py in "${PYTHONS[@]}"; do
      echo "[INFO] $pkg ($plat, python$py) wheel 다운로드 시도..."
      pip download --only-binary=:all: --platform $plat --python-version $py --implementation cp --abi abi3 $pkg -d offline_packages || true
    done
  done
done 