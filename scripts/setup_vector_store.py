import os

index_path = "data/index/codeai.index"

if not os.path.exists("data/index"):
    os.makedirs("data/index")

with open(index_path, "w") as f:
    f.write("(mock) FAISS index initialized\n")

print(f"[OK] 벡터스토어 초기화 완료: {index_path}") 