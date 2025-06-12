#!/bin/bash

# 오프라인 패키지 다운로드 스크립트
# 이 스크립트는 필요한 모든 Python 패키지, 모델, Docker 이미지를 다운로드합니다.

# 오류 발생 시 즉시 중단
set -e

# 작업 디렉토리 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$WORKSPACE_DIR/offline_packages"
DOCKER_IMAGES_DIR="$OFFLINE_DIR/docker-images"
TREE_SITTER_DIR="$OFFLINE_DIR/tree-sitter-languages"

# 디렉토리 생성
mkdir -p "$OFFLINE_DIR"
mkdir -p "$DOCKER_IMAGES_DIR"
mkdir -p "$TREE_SITTER_DIR"

echo "[INFO] 오프라인 패키지 다운로드를 시작합니다..."

# Python 패키지 다운로드
echo "[INFO] Python 패키지 다운로드 중..."
pip download -r "$WORKSPACE_DIR/requirements.txt" -d "$OFFLINE_DIR"

# Tree-sitter 언어 파일 다운로드 및 빌드
echo "[INFO] Tree-sitter CLI 설치 및 언어 파일 빌드 중..."

# Tree-sitter CLI 설치
if ! command -v tree-sitter &> /dev/null; then
    echo "[INFO] Tree-sitter CLI 설치 중..."
    npm install -g tree-sitter-cli
fi

# 지원하는 언어 목록
LANGUAGES=(
    "python:https://github.com/tree-sitter/tree-sitter-python"
    "java:https://github.com/tree-sitter/tree-sitter-java"
    "javascript:https://github.com/tree-sitter/tree-sitter-javascript"
    "typescript:https://github.com/tree-sitter/tree-sitter-typescript"
    "cpp:https://github.com/tree-sitter/tree-sitter-cpp"
    "c:https://github.com/tree-sitter/tree-sitter-c"
)

# 각 언어 파일 빌드
for lang_info in "${LANGUAGES[@]}"; do
    IFS=':' read -r lang_name repo_url <<< "${lang_info}"
    echo "[INFO] ${lang_name} 언어 파일 빌드 중..."
    TEMP_DIR=$(mktemp -d)
    cd "${TEMP_DIR}"
    
    # 특수 처리: typescript
    if [ "${lang_name}" = "typescript" ]; then
        git clone "${repo_url}" .
        for sub in typescript tsx; do
            if [ -d "$sub" ]; then
                cd "$sub"
                tree-sitter generate
                if [ -f "src/parser.c" ]; then
                    if [ -f "src/scanner.c" ]; then
                        gcc -shared -o "${sub}.so" -fPIC src/parser.c src/scanner.c
                    elif [ -f "src/scanner.cc" ]; then
                        g++ -shared -o "${sub}.so" -fPIC src/parser.c src/scanner.cc
                    else
                        gcc -shared -o "${sub}.so" -fPIC src/parser.c
                    fi
                    cp "${sub}.so" "${TREE_SITTER_DIR}/"
                fi
                cd ..
            fi
        done
        cd - > /dev/null
        rm -rf "${TEMP_DIR}"
        continue
    fi

    # 특수 처리: cpp (tree-sitter-c도 필요)
    if [ "${lang_name}" = "cpp" ]; then
        git clone "${repo_url}" .
        git clone https://github.com/tree-sitter/tree-sitter-c tree-sitter-c
        tree-sitter generate
        if [ -f "src/parser.c" ]; then
            if [ -f "src/scanner.c" ]; then
                gcc -shared -o "${lang_name}.so" -fPIC src/parser.c src/scanner.c
            elif [ -f "src/scanner.cc" ]; then
                g++ -shared -o "${lang_name}.so" -fPIC src/parser.c src/scanner.cc
            else
                gcc -shared -o "${lang_name}.so" -fPIC src/parser.c
            fi
            cp "${lang_name}.so" "${TREE_SITTER_DIR}/"
        fi
        cd - > /dev/null
        rm -rf "${TEMP_DIR}"
        continue
    fi

    # 일반 언어 처리
    git clone "${repo_url}" .
    tree-sitter generate
    if [ -f "src/parser.c" ]; then
        if [ -f "src/scanner.c" ]; then
            gcc -shared -o "${lang_name}.so" -fPIC src/parser.c src/scanner.c
        elif [ -f "src/scanner.cc" ]; then
            g++ -shared -o "${lang_name}.so" -fPIC src/parser.c src/scanner.cc
        else
            gcc -shared -o "${lang_name}.so" -fPIC src/parser.c
        fi
        cp "${lang_name}.so" "${TREE_SITTER_DIR}/"
    fi
    cd - > /dev/null
    rm -rf "${TEMP_DIR}"
done

# Docker 이미지 다운로드
echo "[INFO] Docker 이미지 다운로드 중..."
docker pull python:3.11-slim
docker pull neo4j:5.15.0
docker pull redis:7.2.4

# Docker 이미지 저장
docker save python:3.11-slim | gzip > "${DOCKER_IMAGES_DIR}/python.tar.gz"
docker save neo4j:5.15.0 | gzip > "${DOCKER_IMAGES_DIR}/neo4j.tar.gz"
docker save redis:7.2.4 | gzip > "${DOCKER_IMAGES_DIR}/redis.tar.gz"

echo "[INFO] Docker 환경에서 오프라인 패키지 다운로드 완료"

echo "[INFO] 오프라인 설치 준비가 완료되었습니다."

echo "===== 오프라인 설치 가이드 ====="
echo "1. offline_packages/와 offline_packages/docker-images/ 폴더를 오프라인 서버로 복사하세요."
echo "2. 오프라인 서버에서 아래 명령을 순서대로 실행하세요:"
echo ""
echo "   # (선택) 가상환경 생성 및 활성화"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo ""
echo "   # 오프라인 Docker 이미지 로드"
echo "   for image in offline_packages/docker-images/*.tar.gz; do"
echo "     docker load < \"\$image\""
echo "   done"
echo ""
echo "   # 오프라인 패키지 설치"
echo "   pip install --no-index --find-links=offline_packages -r requirements.txt"
echo ""
echo "   # Tree-sitter 언어 파일 설치"
echo "   mkdir -p build"
echo "   cp offline_packages/tree-sitter-languages/*.so build/"
echo ""
echo "3. Docker Compose로 서비스 시작"
echo "   docker-compose up -d"

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