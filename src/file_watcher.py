import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from typing import Callable, List, Set, Optional
import threading
from dataclasses import dataclass

@dataclass
class FileChangeEvent:
    """파일 변경 이벤트"""
    path: str
    event_type: str  # created, modified, deleted, moved
    timestamp: float

class CodebaseWatcher(FileSystemEventHandler):
    """코드베이스 파일 변경 모니터링"""
    
    def __init__(self, 
                 project_path: str,
                 on_change_callback: Callable[[FileChangeEvent], None],
                 file_extensions: Optional[List[str]] = None):
        self.project_path = Path(project_path)
        self.callback = on_change_callback
        self.file_extensions = file_extensions or ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go']
        self.pending_changes: Set = set()
        self.debounce_delay = 1.0  # 1초 디바운스
        self.debounce_task = None
        self.exclude_patterns = {
            '__pycache__', '.git', '.vscode', 'node_modules', 
            '.pytest_cache', '.mypy_cache', 'dist', 'build'
        }
    
    def _should_process_file(self, file_path: str) -> bool:
        path = Path(file_path)
        for part in path.parts:
            if part in self.exclude_patterns:
                return False
        if self.file_extensions and path.suffix not in self.file_extensions:
            return False
        return True
    
    def on_modified(self, event):
        if not event.is_directory and self._should_process_file(event.src_path):
            self._schedule_callback(event.src_path, "modified")
    
    def on_created(self, event):
        if not event.is_directory and self._should_process_file(event.src_path):
            self._schedule_callback(event.src_path, "created")
    
    def on_deleted(self, event):
        if not event.is_directory and self._should_process_file(event.src_path):
            self._schedule_callback(event.src_path, "deleted")
    
    def on_moved(self, event):
        if not event.is_directory:
            if self._should_process_file(event.src_path):
                self._schedule_callback(event.src_path, "deleted")
            if self._should_process_file(event.dest_path):
                self._schedule_callback(event.dest_path, "created")
    
    def _schedule_callback(self, file_path: str, event_type: str):
        self.pending_changes.add((file_path, event_type))
        if self.debounce_task:
            self.debounce_task.cancel()
        self.debounce_task = threading.Timer(
            self.debounce_delay, 
            self._process_pending_changes
        )
        self.debounce_task.start()
    
    def _process_pending_changes(self):
        changes_to_process = self.pending_changes.copy()
        self.pending_changes.clear()
        for file_path, event_type in changes_to_process:
            try:
                event = FileChangeEvent(
                    path=file_path,
                    event_type=event_type,
                    timestamp=time.time()
                )
                self.callback(event)
            except Exception as e:
                print(f"파일 변경 콜백 오류: {e}")

class AutoIndexingSystem:
    """자동 인덱싱 시스템"""
    
    def __init__(self, project_path: str, rag_system, ai_agent):
        self.project_path = project_path
        self.rag_system = rag_system
        self.ai_agent = ai_agent
        self.observer = None
        self.watcher = None
        self.is_running = False
        self.change_stats = {
            "files_changed": 0,
            "files_indexed": 0,
            "last_update": None
        }
    
    def start_monitoring(self):
        if self.is_running:
            return
        print(f"파일 모니터링 시작: {self.project_path}")
        self.watcher = CodebaseWatcher(
            self.project_path,
            self._on_file_change
        )
        self.observer = Observer()
        self.observer.schedule(self.watcher, self.project_path, recursive=True)
        self.observer.start()
        self.is_running = True
    
    def stop_monitoring(self):
        if not self.is_running:
            return
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.is_running = False
        print("파일 모니터링 중지")
    
    def _on_file_change(self, event: FileChangeEvent):
        print(f"파일 변경 감지: {event.event_type} - {event.path}")
        try:
            if event.event_type in ["created", "modified"]:
                self.rag_system.index_file(event.path)
                self.change_stats["files_indexed"] += 1
                asyncio.create_task(self._notify_agent_of_change(event))
            elif event.event_type == "deleted":
                self.rag_system.remove_file_from_index(event.path)
            self.change_stats["files_changed"] += 1
            self.change_stats["last_update"] = event.timestamp
        except Exception as e:
            print(f"파일 변경 처리 오류: {e}")
    
    async def _notify_agent_of_change(self, event: FileChangeEvent):
        try:
            if self._is_important_file(event.path):
                message = f"파일이 {event.event_type}되었습니다: {event.path}. 이 변경사항을 검토해주세요."
                await self.ai_agent.process_message(message)
        except Exception as e:
            print(f"에이전트 알림 오류: {e}")
    
    def _is_important_file(self, file_path: str) -> bool:
        important_patterns = [
            'main.py', 'app.py', '__init__.py',
            'requirements.txt', 'setup.py', 'Dockerfile',
            'README.md', 'config.py'
        ]
        file_name = Path(file_path).name.lower()
        return any(pattern in file_name for pattern in important_patterns) 