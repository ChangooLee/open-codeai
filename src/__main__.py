from indexer import index_project
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src <project_path>")
        exit(1)
    index_project(sys.argv[1]) 