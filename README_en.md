한국어 | [English](README_en.md)

# Open CodeAI

![License](https://img.shields.io/github/license/ChangooLee/open-codeai)

Open CodeAI is an open-source AI code assistant for large-scale projects and air-gapped environments. It provides secure and contextual AI code support while maintaining data privacy and security.

This project is licensed under the [MIT License](LICENSE).

---

[Table of Contents](#table-of-contents)

# Quick Start Guide by OS

## macOS

1. **Install Required Tools**
   - Install [Homebrew](https://brew.sh/)
   - Install Python 3.10+: `brew install python`
   - Install Docker Desktop: [Official Download](https://www.docker.com/products/docker-desktop/)
2. **Copy Offline Packages/Models**
   - Copy `offline_packages/`, `data/models/` folders
3. **Install and Run**
   ```bash
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
2. **Copy Offline Packages/Models**
   - Copy `offline_packages\`, `data\models\` folders
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
2. **Copy Offline Packages/Models**
   - Copy `offline_packages/`, `data/models/` folders
3. **Install and Run**
   ```bash
   chmod +x install.sh
   ./install.sh --offline
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
- **Qwen2.5-Coder-based LLM**: Code generation/completion on par with Cursor AI
- **FAISS + Graph DB (Neo4j/NetworkX)**: Semantic/relationship search for large codebases
- **Continue.dev Integration**: Use directly from VSCode/JetBrains
- **Automated Setup/Configuration**: config.yaml → .env, auto-detects offline packages/models
- **Supports Containerless/Sharding/Quantization Modes**

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