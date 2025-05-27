from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
import faiss
import os

class CodeEmbedder:
    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", index_dir="./data/index"):
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        Settings.embed_model = self.embed_model
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.dimension = 1024  # BGE-large 기준
        self.faiss_index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.vector_index = VectorStoreIndex([], storage_context=self.storage_context)

    def add_documents(self, docs):
        for doc in docs:
            self.vector_index.insert(doc)

    def save(self):
        faiss.write_index(self.faiss_index, f"{self.index_dir}/faiss_index") 