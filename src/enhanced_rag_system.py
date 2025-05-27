import os
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

# 가벼운 RAG 시스템 베이스 클래스 (실제 구현 필요)
class CodeRAGSystem:
    def search_code(self, query):
        return []

class EnhancedRAGSystem(CodeRAGSystem):
    """동적 업데이트를 지원하는 향상된 RAG 시스템"""
    
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.metadata_db_path = self.index_dir / "metadata.db"
        self._init_metadata_db()
        self.file_hashes = {}
        self._load_file_hashes()

    def _init_metadata_db(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    indexed_at TIMESTAMP NOT NULL,
                    file_size INTEGER,
                    line_count INTEGER,
                    language TEXT,
                    last_modified TIMESTAMP
                )
            """)

    def _load_file_hashes(self):
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                cursor = conn.execute("SELECT file_path, file_hash FROM file_metadata")
                self.file_hashes = dict(cursor.fetchall())
        except Exception as e:
            print(f"파일 해시 로드 오류: {e}")
            self.file_hashes = {}

    def _calculate_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def needs_reindexing(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        current_hash = self._calculate_file_hash(file_path)
        stored_hash = self.file_hashes.get(file_path)
        return current_hash != stored_hash

    def index_file(self, file_path: str, force: bool = False) -> bool:
        if not force and not self.needs_reindexing(file_path):
            print(f"파일 변경 없음, 스킵: {file_path}")
            return False
        print(f"파일 인덱싱: {file_path}")
        self._update_file_metadata(file_path)
        return True

    def remove_file_from_index(self, file_path: str):
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (file_path,))
            self.file_hashes.pop(file_path, None)
        except Exception as e:
            print(f"파일 제거 오류 {file_path}: {e}")

    def _update_file_metadata(self, file_path: str):
        try:
            file_stat = os.stat(file_path)
            file_hash = self._calculate_file_hash(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = len(f.readlines())
            language = Path(file_path).suffix[1:]
            with sqlite3.connect(self.metadata_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO file_metadata 
                    (file_path, file_hash, indexed_at, file_size, line_count, language, last_modified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_path,
                    file_hash,
                    datetime.now().isoformat(),
                    file_stat.st_size,
                    line_count,
                    language,
                    datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                ))
            self.file_hashes[file_path] = file_hash
        except Exception as e:
            print(f"파일 메타데이터 업데이트 오류: {e}")

    def get_project_statistics(self):
        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                file_stats = conn.execute("SELECT COUNT(*) as total_files FROM file_metadata").fetchone()
                return {"file_statistics": file_stats, "last_updated": datetime.now().isoformat()}
        except Exception as e:
            print(f"통계 조회 오류: {e}")
            return {} 