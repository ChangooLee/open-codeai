#!/usr/bin/env python3
"""
Tree-sitter 언어 파일 빌드 스크립트
"""
import os
import subprocess
from pathlib import Path
import shutil
import sys

def run_command(cmd, cwd=None):
    """명령어 실행"""
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def clone_repo(repo_url, target_dir):
    """저장소 클론"""
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    success, output = run_command(['git', 'clone', '--depth', '1', repo_url, target_dir])
    return success

def build_language(lang_name, repo_url):
    """언어 파일 빌드"""
    print(f"\nBuilding {lang_name}...")
    
    # 작업 디렉토리 설정
    build_dir = Path('build')
    build_dir.mkdir(exist_ok=True)
    
    # 임시 디렉토리
    temp_dir = build_dir / f"tree-sitter-{lang_name}"
    
    # 저장소 클론
    if not clone_repo(repo_url, temp_dir):
        print(f"Failed to clone {lang_name} repository")
        return False
    
    # 빌드
    success, output = run_command(['tree-sitter', 'generate'], cwd=temp_dir)
    if not success:
        print(f"Failed to generate {lang_name} grammar: {output}")
        return False
    
    # 컴파일
    success, output = run_command(['tree-sitter', 'build-wasm'], cwd=temp_dir)
    if not success:
        print(f"Failed to build {lang_name} wasm: {output}")
        return False
    
    # 파일 복사
    source_file = temp_dir / f"{lang_name}.so"
    target_file = build_dir / f"{lang_name}.so"
    if source_file.exists():
        shutil.copy2(source_file, target_file)
        print(f"Successfully built {lang_name}")
        return True
    else:
        print(f"Built file not found for {lang_name}")
        return False

def main():
    """메인 함수"""
    # 언어 정의
    languages = {
        'python': 'https://github.com/tree-sitter/tree-sitter-python.git',
        'java': 'https://github.com/tree-sitter/tree-sitter-java.git',
        'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript.git',
        'typescript': 'https://github.com/tree-sitter/tree-sitter-typescript.git',
        'cpp': 'https://github.com/tree-sitter/tree-sitter-cpp.git',
        'c': 'https://github.com/tree-sitter/tree-sitter-c.git'
    }
    
    # tree-sitter 설치 확인
    success, output = run_command(['which', 'tree-sitter'])
    if not success:
        print("Error: tree-sitter not found. Please install it first.")
        print("Installation guide: https://tree-sitter.github.io/tree-sitter/creating-parsers#installation")
        sys.exit(1)
    
    # 각 언어 빌드
    success_count = 0
    for lang_name, repo_url in languages.items():
        if build_language(lang_name, repo_url):
            success_count += 1
    
    print(f"\nBuild completed: {success_count}/{len(languages)} languages built successfully")
    
    # 임시 디렉토리 정리
    build_dir = Path('build')
    for item in build_dir.glob('tree-sitter-*'):
        if item.is_dir():
            shutil.rmtree(item)

if __name__ == '__main__':
    main() 