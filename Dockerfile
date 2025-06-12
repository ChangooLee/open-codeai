FROM python:3.12-slim

WORKDIR /app

COPY offline_packages /offline_packages

COPY requirements.txt ./

# 빌드 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends make gcc g++ python3-dev curl && \
    pip install --upgrade pip && \
    pip install --no-index --find-links=/offline_packages -r requirements.txt && \
    apt-get remove -y make gcc g++ python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Tree-sitter 빌드 환경 설정
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Tree-sitter CLI 설치 (npm을 통한 설치)
RUN npm install -g tree-sitter-cli

# Tree-sitter 언어 파일 빌드
COPY scripts/build_tree_sitter.py /app/scripts/
RUN chmod +x /app/scripts/build_tree_sitter.py \
    && mkdir -p /app/build \
    && python /app/scripts/build_tree_sitter.py

COPY . .

EXPOSE 8800
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8800"] 