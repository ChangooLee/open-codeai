FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
COPY requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"] 