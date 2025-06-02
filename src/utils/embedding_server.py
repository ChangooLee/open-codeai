import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/codebert-base")
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("DEVICE", "auto") != "cpu" else "cpu"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

class EmbeddingRequest(BaseModel):
    texts: list[str]

@app.post("/embedding")
def embedding(req: EmbeddingRequest):
    with torch.no_grad():
        inputs = tokenizer(req.texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
    return {"embeddings": embeddings} 