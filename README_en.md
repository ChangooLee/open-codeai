한국어 | [English](README_en.md)

# Open CodeAI

![License](https://img.shields.io/github/license/ChangooLee/open-codeai)

Open CodeAI is an open-source AI code assistant for large-scale projects and air-gapped environments. It provides secure and contextual AI code support while maintaining data privacy and security.

This project is licensed under the [MIT License](LICENSE).

---

[Table of Contents](#table-of-contents)

# Quick Start Guide by OS

## Installation Options

Open CodeAI can be installed in both online and offline environments.

### Virtual Environment Setup
It is recommended to create and activate a Python virtual environment before installation:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt)
.\.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate
```

#### Additional Windows Setup Guide
1. **PowerShell Execution Policy**
   - If script execution is blocked in PowerShell, run as administrator:
   ```powershell
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Visual Studio Build Tools**
   - Required for some package installations
   - Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Select "C++ build tools" workload during installation

3. **Python Path Configuration**
   - Verify Python and pip are in system PATH
   - Check in Command Prompt:
   ```cmd
   python --version
   pip --version
   ```

4. **Virtual Environment Troubleshooting**
   - If activation fails:
     - PowerShell: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`
     - Command Prompt: Run as administrator
   - If path contains non-ASCII characters, try moving to an ASCII-only path

### Online Installation (Default)
For online environments, use the following command:
```bash
./scripts/install.sh
```

### Offline Installation
For offline environments, you need to download the required packages first:

1. Download packages in an online environment:
```bash
./scripts/download_offline_packages.sh
```

2. Install with downloaded packages:
```bash
./scripts/install.sh --offline
```

## macOS

1. **Install Required Tools**
   - Install [Homebrew](https://brew.sh/)
   - Install Python 3.10+: `brew install python`
   - Install Docker Desktop: [Official Download](https://www.docker.com/products/docker-desktop/)
2. **Project Clone**
   - Clone this repository to get the required offline_packages/ folder automatically:
   ```bash
   git clone https://github.com/ChangooLee/open-codeai.git
   cd open-codeai
   ```
3. **Install and Run**
   ```bash
   cd open-codeai/scripts   # Make sure you are in the scripts folder!
   chmod +x install.sh
   ./install.sh --offline
   ./start.sh
   ./index.sh /Users/yourname/Workspace/yourproject
   ```
4. **Permissions/Security Issues**
   - Grant execute permission: `chmod +x *.sh`
   - Recommended to run in terminal (zsh, bash)
   - Ensure Docker Desktop is running

## Windows

1. **Install Required Tools**
   - Install [Python 3.10+](https://www.python.org/downloads/windows/) (check Add to PATH)
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. **Project Clone**
   - Clone this repository to get the required offline_packages/ folder automatically:
   ```bash
   git clone https://github.com/ChangooLee/open-codeai.git
   cd open-codeai
   ```
3. **Install and Run**
   - In PowerShell:
     ```powershell
     .\install.bat
     .\start.bat
     .\index.bat C:\Users\yourname\Workspace\yourproject
     ```
   - Or in WSL (recommended): follow Linux instructions in Ubuntu environment
4. **Path/Permission/Korean Path Caution**
   - Avoid Korean/space/special characters in paths
   - Run as administrator if needed (PowerShell "Run as Administrator")
   - Ensure Docker Desktop is running

## Linux (Ubuntu, etc.)

1. **Install Required Tools**
   - Install Python 3.10+: `sudo apt install python3 python3-venv python3-pip`
   - Install Docker: `curl -fsSL https://get.docker.com | sh`
2. **Project Clone**
   - Clone this repository to get the required offline_packages/ folder automatically:
   ```bash
   git clone https://github.com/ChangooLee/open-codeai.git
   cd open-codeai
   ```
3. **Install and Run**
   ```bash
   cd open-codeai   # Make sure you are in the project root!
   chmod +x scripts/install.sh
   ./scripts/install.sh --offline
   ./start.sh
   ./index.sh /home/yourname/yourproject
   ```
4. **Permissions/Security Issues**
   - Grant execute permission: `chmod +x *.sh`
   - Add to Docker group: `sudo usermod -aG docker $USER`
   - Ensure Docker Desktop/engine is running

## Common Notes

- For offline installation, copy all packages/models/docker images in advance
- Docker Desktop/engine must be running
- Refer to OS-specific messages for environment/path issues
- Avoid using Korean/space/special characters in paths

---

## ✨ Key Features
- **Fully Offline/Air-gapped Installation**: All features available without internet
- **Qwen2.5-Coder based LLM**: Code generation/completion at Cursor AI level (vLLM endpoint auto-integrated)
- **FAISS + Graph DB (Neo4j/NetworkX)**: Semantic/relationship search for large codebases
- **Continue.dev Integration**: Use directly from VSCode/JetBrains
- **Automated Setup/Configuration**: config.yaml → .env, auto-detects offline packages/models
- **Supports Containerless/Sharding/Quantization Modes**
- **vLLM engine**: Assumes endpoint is available, no separate installation required.
- **Custom code embedding model support**
- Real-time indexing status and next action guidance
- Automatic indexing of `README.md` and `README_en.md`

---

## 🏁 Quick Start (Offline/Air-gapped)

### 1. Prepare Dependencies/Models
- Pre-copy Python wheel files (.whl) to `offline_packages/`
- Pre-copy LLM/embedding/graph model files to `data/models/`
- (Optional) Copy Docker image tar files to `docker-images/`

### 2. Install and Initialize
```bash
# 1. Extract and move
$ tar -xzf open-codeai-*.tar.gz && cd open-codeai
# 2. Offline install
$ ./install.sh --offline
# 3. (Auto-detects pre-copied models/packages)
```

### 3. Run Server and Index Project
```bash
# Run server
$ ./start.sh
# Index project (first time only)
$ ./index.sh /path/to/your/project
```

### 4. Use directly in VSCode with Continue extension

---

## ⚙️ Automated Configuration & Key Options

- Manage all settings in **config.yaml** (models/DB/performance/modes, etc.)
- Use `scripts/generate_env.py` to auto-generate .env (run after config.yaml changes)
- Example options:
```yaml
llm:
  main_model:
    name: "qwen2.5-coder-32b"
    path: "./data/models/qwen2.5-coder-32b"
    use_vllm: true
    quantize: "4bit"   # none, 4bit, 8bit
    device: "auto"
database:
  graph:
    type: "networkx"   # neo4j or networkx
    auto_select: true
  vector:
    sharding: true
performance:
  gpu:
    enable: true
    mixed_precision: true
```

---

## 🧩 Various Modes/Extensibility

- **Offline install**: Only need offline_packages/, data/models/ folders, no internet required
- **Containerless**: Set `database.graph.type: networkx` in config.yaml, no Neo4j/Redis
- **Minimal mode**: `--minimal` flag, disables monitoring/Continue.dev
- **Quantization/Sharding**: `llm.main_model.quantize: 4bit`, `database.vector.sharding: true`
- **No Neo4j**: `./install.sh --no-neo4j` or set networkx in config

---

## 🛠️ Key Scripts/Automation Tools

- `install.sh` : Offline/online auto-install, auto-detects models/packages/docker images
- `scripts/generate_env.py` : Auto-convert config.yaml → .env
- `start.sh` : Unified server/docker/venv runner
- `index.sh` : Project indexing automation
- `scripts/verify_installation.py` : Installation/environment verification

---

## 🧑‍💻 Development/Extension Guide

- **Replace/add models**: Just change main_model/embedding_model path in config.yaml
- **Expand DB/index structure**: Adjust graph/vector options in config.yaml, supports sharding/containerless
- **Custom Continue.dev commands/prompts**: `~/.continue/config.json` auto-generated, can be edited
- **Code/config refactoring**: See src/, configs/, scripts/ structure

---

## ❓ FAQ & Troubleshooting

- **Q. Offline install not working!**
  - Check files in offline_packages/, data/models/
  - Check logs/errors when running install.sh
- **Q. Want to run without Neo4j**
  - `./install.sh --no-neo4j` or set `database.graph.type: networkx` in config.yaml
- **Q. Model/package version mismatch**
  - Check config.yaml, requirements.txt, offline_packages/ versions
- **Q. Indexing is slow/out of memory**
  - Adjust parallel_workers, memory_limit_gb, chunk_size in config.yaml
- **Q. Getting authentication errors with vLLM?**
  - Make sure the `.env` file is in the project root and the API Key is correct
  - You must use the Authorization header (`Bearer ...`) for authentication; do NOT put api_key in the body
  - Ensure there are no leftover config/env files in configs/ or subfolders

---

## 📸 Screenshots/Architecture
(Architecture diagram, VSCode integration, indexing/search examples, etc. to be added)

---

## 📝 Reference/Contributing
- [Official Docs/Wiki](https://github.com/ChangooLee/open-codeai/wiki)
- [Issue/Contributing Guide](https://github.com/ChangooLee/open-codeai/CONTRIBUTING.md)
- [Continue.dev Official](https://continue.dev/)

---

Open CodeAI provides the best code AI experience even in large enterprises, public institutions, and air-gapped environments!

---

MIT License

Copyright (c) 2025 [LEE CHANGOO/Samsungcard]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## ⚙️ Environment Setup (.env) & DB Info

- Use a `.env` or `.env.example` file in the project root.
- **Vector DB/Graph DB** require no separate installation or configuration—they are managed internally. Users only need to set the port (VECTOR_DB_PORT, GRAPH_DB_PORT) if needed.
- All data/model paths are managed automatically; you do not need to set them.

### .env.example
```env
# Open CodeAI .env example file
# Copy this file to .env for your setup (cp .env.example .env)

# === LLM/vLLM Engine ===
VLLM_ENDPOINT=http://localhost:8800/v1
VLLM_API_KEY=your-vllm-api-key
VLLM_MODEL_ID=Qwen2.5-Coder-32B-Instruct

# === Server ===
HOST=0.0.0.0
PORT=8800
LOG_LEVEL=INFO

# === Project Info ===
PROJECT_NAME=open-codeai
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=True

# === Vector DB (FAISS, etc.) ===
# No installation/config needed, managed internally
VECTOR_DB_PORT=9000  # (set port if needed, default 9000)

# === Graph DB (NetworkX/Neo4j) ===
# No installation/config needed, managed internally
GRAPH_DB_PORT=7687  # (set port if needed, default 7687)

# === Other ===
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
GPU_MEMORY_FRACTION=0.7
USE_MIXED_PRECISION=True
API_KEY=open-codeai-local-key
```

## Code Embedding Model Recommendation

### Recommended Models
- **BAAI/bge-code-v1.5** (recommended, multi-language, dimension=1024)
- microsoft/codebert-base
- Salesforce/codet5-base

### How to Configure
- Add to your `.env` file:
  ```
  EMBEDDING_MODEL_NAME=BAAI/bge-code-v1.5
  EMBEDDING_MODEL_DIM=1024
  ```
- Or set directly in `src/config.py`

## Indexing & Status

- Start indexing: `/admin/index-project` (POST)
- Check status: `/status` (GET)
  - While indexing: `"indexing": true`, progress/message/next_actions provided
  - When done: `"indexing": false`, "You can now search/ask about the codebase" guidance

## README Auto-Indexing
- Both `README.md` and `README_en.md` are always included in the index.
- Keep these files up to date for best LLM understanding of your project.

## Example API Response
```json
{
  "indexing": true,
  "indexing_progress": 120,
  "indexing_total": 500,
  "message": "Indexing in progress... (120/500)",
  "next_actions": [
    {"action": "wait", "description": "Please wait until indexing is complete."}
  ]
}
```

## Contributing & Support
- [GitHub Issues](https://github.com/your-repo)

## 🧩 Embedding Model (microsoft/codebert-base) Download Guide

- The default embedding model is **microsoft/codebert-base**. This is a public model and can be downloaded without authentication.
- Download the model and place it in the `offline_packages/codebert-base` folder.

### Download Steps
1. Run the following commands in your terminal:
   ```bash
   python3 -m pip install huggingface_hub
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/codebert-base', local_dir='offline_packages/codebert-base', local_dir_use_symlinks=False)"
   ```
2. After download, ensure the `offline_packages/codebert-base` folder exists and contains the model files. 

## Code Embedding/Analysis per Project (Docker)

Open CodeAI runs in a Docker container, and you can specify the project root directory and command when starting the service.

### Usage

```bash
# [project_directory] [command] format (command: start/stop/restart/status/logs/help)
./scripts/start.sh /path/to/your/project start
./scripts/start.sh /path/to/your/project
./scripts/start.sh start  # Use current directory as project
```
- If the first argument is an existing directory, it is used as the project path and the second argument is the command.
- If the first argument is not a directory, it is treated as the command and the current directory is used as the project path.
- If no command is given, the default is `start`.

- The specified path will be mounted to `/workspace` inside the Docker container.
- The FastAPI server will index/embed code under `/workspace`.
- To analyze multiple projects, simply restart start.sh with a different path.

### Example

```bash
./scripts/start.sh ~/projects/my-awesome-project start
./scripts/start.sh ~/projects/my-awesome-project
./scripts/start.sh restart  # Restart with current directory as project
```

> **Note:**
> - You cannot change the mount path while the container is running. To switch projects, stop the container and restart with a new path.
> - For production, only mount necessary directories for security. 

## 📝 Advanced Logging System

- **Loguru-based advanced logging**: All API requests, responses, errors, and performance metrics are automatically logged.
- Log files: `logs/opencodeai.log` (general), `logs/error.log` (errors), `logs/performance.log` (performance)
- Logs are automatically rotated, compressed, and retained (up to 30/90/7 days)
- Log level, path, and retention can be configured in config.yaml

## 🗃️ Vector DB/Graph DB/Embedding System

- **Vector DB (FAISS)**: Code embeddings are stored/searched in a FAISS (HNSW) index, with index files automatically saved under `data/vector_index/<project_name>/`.
- **Graph DB (Neo4j/NetworkX)**: Code structure (files, functions, classes, dependencies, call relations, etc.) is stored in a graph DB. Neo4j (default) or NetworkX (containerless) is automatically selected, with graph files saved under `data/graph_db/<project_name>/`.
- **Embedding server/model**: Uses Huggingface-based embedding models (e.g., `microsoft/codebert-base`), configurable via `.env`/`config.yaml`. Embedding API (`/embedding`) allows direct text/code embedding.

## 🛠️ start.sh, install.sh Key Features

- `start.sh`: Automatically sets up project-specific data directories, vector/graph DB paths, supports various commands (`start`, `stop`, `status`, `logs`), and includes log/status checking.
- `install.sh`: Fully automates offline package installation, Docker image build, Neo4j/Redis container setup, .env auto-generation, data directory creation, and GPU/CPU environment detection. 

### Troubleshooting Installation

1. **Environment Variables**
   - Create `.env` file in project root:
   ```env
   PROJECT_PATH=/path/to/your/workspace
   PROJECT_BASENAME=open-codeai
   ```
   - Or set environment variables directly:
   ```bash
   export PROJECT_PATH=/path/to/your/workspace
   export PROJECT_BASENAME=open-codeai
   ```

2. **Permission Issues**
   - Check script execution permissions: `chmod +x *.sh`
   - Verify Docker group permissions: `sudo usermod -aG docker $USER`

3. **Docker Desktop Status**
   - Ensure Docker Desktop is running
   - Check Docker daemon status: `docker info` 