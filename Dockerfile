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

COPY . .

EXPOSE 8800
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8800"] 