from code_parser import CodeParser
from embedder import CodeEmbedder
from llama_index.core import Document
from pathlib import Path


def index_project(project_path, embed_model_name="BAAI/bge-large-en-v1.5", index_dir="./data/index"):
    parser = CodeParser()
    embedder = CodeEmbedder(embed_model_name=embed_model_name, index_dir=index_dir)
    code_files = parser.walk_project(project_path)
    docs = []
    for file_path in code_files:
        blocks = parser.parse_file(file_path)
        for block in blocks:
            docs.append(Document(
                text=block.code,
                metadata={
                    'file_path': block.file_path,
                    'type': block.type,
                    'name': block.name,
                    'start_line': block.start_line,
                    'end_line': block.end_line
                }
            ))
    embedder.add_documents(docs)
    embedder.save()
    print(f"Indexed {len(docs)} code blocks from {len(code_files)} files.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.indexer <project_path>")
        exit(1)
    index_project(sys.argv[1]) 