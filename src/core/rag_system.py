"""
Open CodeAI - RAG (Retrieval-Augmented Generation) 시스템
벡터 DB와 그래프 DB를 활용한 코드베이스 검색 및 분석
"""
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import hashlib
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import traceback

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from neo4j import GraphDatabase
    import networkx as nx
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

import sqlite3
from dataclasses import dataclass, asdict
from ..config import settings
from ..utils.logger import get_logger, log_performance
from .code_analyzer import get_code_analyzer
from .llm_manager import get_llm_manager

logger = get_logger(__name__)

@dataclass
class CodeChunk:
    """코드 청크 정보"""
    id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # function, class, import, comment, other
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None

@dataclass
class SearchResult:
    """검색 결과"""
    chunk: CodeChunk
    similarity_score: float
    relevance_score: float = 0.0
    graph_connections: List[Dict] = None

class VectorDatabase:
    """FAISS 기반 벡터 데이터베이스"""
    
    def __init__(self, dimension: int = None, index_path: str = ""):
        # EMBEDDING_MODEL_DIM 환경변수에서 차원 읽기, 기본값 768
        self.dimension = int(os.getenv("EMBEDDING_MODEL_DIM", 768)) if dimension is None else dimension
        self.index_path: str = index_path or settings.VECTOR_INDEX_PATH
        self.index = None
        self.chunk_map: dict[str, CodeChunk] = {}  # chunk_id -> CodeChunk
        self.file_index: dict[str, list[str]] = {}  # file_path -> List[chunk_id]
        self._lock = threading.Lock()
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """인덱스 초기화"""
        try:
            if HAS_FAISS:
                # HNSW 인덱스 사용 (대용량 데이터에 적합)
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
                
                # 기존 인덱스 로드 시도
                if os.path.exists(f"{self.index_path}/vector.index"):
                    self._load_index()
                    
                logger.info(f"벡터 DB 초기화 완료 - 차원: {self.dimension}")
            else:
                logger.warning("FAISS가 설치되지 않았습니다. 더미 벡터 DB를 사용합니다.")
                
        except Exception as e:
            logger.error(f"벡터 DB 초기화 실패: {e}")
    
    def _load_index(self) -> None:
        """저장된 인덱스 로드"""
        try:
            if not HAS_FAISS:
                return
                
            index_file = f"{self.index_path}/vector.index"
            metadata_file = f"{self.index_path}/metadata.json"
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                self.index = faiss.read_index(index_file)
                
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # 청크 매핑 정보 복원
                for chunk_data in metadata.get('chunks', []):
                    chunk = CodeChunk(**chunk_data)
                    self.chunk_map[chunk.id] = chunk
                    
                    if chunk.file_path not in self.file_index:
                        self.file_index[chunk.file_path] = []
                    self.file_index[chunk.file_path].append(chunk.id)
                
                logger.success(f"기존 벡터 인덱스 로드 완료: {len(self.chunk_map)} 청크")
                
        except Exception as e:
            logger.error(f"인덱스 로드 실패: {e}")
    
    def _save_index(self) -> None:
        """인덱스 저장"""
        try:
            if not HAS_FAISS or not self.index:
                return
                
            os.makedirs(self.index_path, exist_ok=True)
            
            # FAISS 인덱스 저장
            index_file = f"{self.index_path}/vector.index"
            faiss.write_index(self.index, index_file)
            
            # 메타데이터 저장
            metadata = {
                'chunks': [asdict(chunk) for chunk in self.chunk_map.values()],
                'dimension': self.dimension,
                'total_chunks': len(self.chunk_map),
                'last_updated': datetime.now().isoformat()
            }
            
            metadata_file = f"{self.index_path}/metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            logger.info(f"벡터 인덱스 저장 완료: {len(self.chunk_map)} 청크")
            
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {e}")
    
    async def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """청크를 벡터 DB에 추가
        
        Args:
            chunks: 추가할 청크 리스트
            
        Returns:
            성공 여부
        """
        if not chunks:
            return False
        
        try:
            # 첫 번째 청크의 임베딩 차원 검증
            first_chunk = chunks[0]
            if not first_chunk.embedding:
                logger.error("임베딩이 없는 청크가 있습니다")
                return False
            
            actual_dim = len(first_chunk.embedding)
            if actual_dim != self.dimension:
                error_msg = f"임베딩 차원 불일치: {actual_dim} != {self.dimension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 모든 청크의 임베딩 차원 검증
            embeddings = []
            valid_chunks = []
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"임베딩이 없는 청크 건너뜀: {chunk.file_path} ({chunk.id})")
                    continue
                
                if len(chunk.embedding) != self.dimension:
                    logger.error(f"임베딩 차원 불일치: {len(chunk.embedding)} != {self.dimension} - {chunk.file_path} ({chunk.id})")
                    continue
                
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.error("유효한 청크가 없습니다")
                return False
            
            # FAISS 인덱스에 추가
            try:
                self.index.add(np.array(embeddings))
            except AssertionError as e:
                if "dimension" in str(e):
                    # 차원 불일치로 인한 오류 - 인덱스 재생성
                    logger.warning("FAISS 인덱스 차원 불일치, 재생성 시도")
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index.add(np.array(embeddings))
                else:
                    raise
                
            # 청크 맵과 파일 인덱스 업데이트
            for chunk in valid_chunks:
                self.chunk_map[chunk.id] = chunk
                if chunk.file_path not in self.file_index:
                    self.file_index[chunk.file_path] = []
                self.file_index[chunk.file_path].append(chunk.id)
            
            return True
            
        except Exception as e:
            logger.error(f"청크 추가 실패: {e}")
            return False
    
    async def search(self, query: str, k: int = 10) -> List[SearchResult]:
        logger.debug(f"[VectorDB] search 진입: query='{query}', k={k}")
        try:
            if not HAS_FAISS or not self.index or len(self.chunk_map) == 0:
                logger.debug("[VectorDB] FAISS 미사용 또는 인덱스 없음")
                return []
            
            # 쿼리 임베딩 생성
            llm_manager = get_llm_manager()
            query_embedding = await llm_manager.generate_embedding(query)
            
            if len(query_embedding) != self.dimension:
                logger.error(f"쿼리 임베딩 차원 불일치: {len(query_embedding)} != {self.dimension}")
                return []
            
            with self._lock:
                # FAISS 검색
                query_vector = np.array([query_embedding], dtype=np.float32)
                similarities, indices = self.index.search(query_vector, min(k, len(self.chunk_map)))
                
                results = []
                chunk_list = list(self.chunk_map.values())
                
                for similarity, idx in zip(similarities[0], indices[0]):
                    if idx >= 0 and idx < len(chunk_list):
                        chunk = chunk_list[idx]
                        result = SearchResult(
                            chunk=chunk,
                            similarity_score=float(similarity)
                        )
                        results.append(result)
            
            logger.debug(f"[VectorDB] 쿼리 임베딩 생성 완료: dim={len(query_embedding)}")
            logger.info(f"[VectorDB] 검색 결과: {len(results)}개 (쿼리: {query[:30]}...)")
            return sorted(results, key=lambda x: x.similarity_score, reverse=True)
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []
    
    def remove_file(self, file_path: str):
        """파일의 모든 청크 제거"""
        try:
            with self._lock:
                if file_path in self.file_index:
                    chunk_ids = self.file_index[file_path]
                    
                    # 청크 매핑에서 제거
                    for chunk_id in chunk_ids:
                        if chunk_id in self.chunk_map:
                            del self.chunk_map[chunk_id]
                    
                    del self.file_index[file_path]
                    
                    logger.info(f"파일 {file_path}의 {len(chunk_ids)} 청크 제거됨")
                    
                    # 인덱스 재구축 필요 (FAISS는 삭제가 어려우므로)
                    if len(chunk_ids) > 0:
                        asyncio.create_task(self._rebuild_index())
                        
        except Exception as e:
            logger.error(f"파일 제거 실패: {e}")
    
    async def _rebuild_index(self) -> None:
        """인덱스 재구축"""
        try:
            if not HAS_FAISS:
                return
                
            logger.info("벡터 인덱스 재구축 시작...")
            
            with self._lock:
                # 새 인덱스 생성
                new_index = faiss.IndexHNSWFlat(self.dimension, 32)
                new_index.hnsw.efConstruction = 200
                new_index.hnsw.efSearch = 50
                
                # 모든 유효한 임베딩 추가
                embeddings = []
                for chunk in self.chunk_map.values():
                    if chunk.embedding and len(chunk.embedding) == self.dimension:
                        embeddings.append(chunk.embedding)
                
                if embeddings:
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    new_index.add(embeddings_array)
                
                self.index = new_index
                self._save_index()
            
            logger.success("벡터 인덱스 재구축 완료")
            
        except Exception as e:
            logger.error(f"인덱스 재구축 실패: {e}")

    def search_file_by_name(self, filename: str) -> list:
        """파일명으로 전체 경로 후보를 반환"""
        # vector DB: file_index의 key(전체 경로)에서 파일명 일치/포함 검색
        matches = [path for path in self.file_index.keys() if os.path.basename(path) == filename or filename in os.path.basename(path)]
        return matches

class GraphDatabase:
    """Neo4j/NetworkX 기반 그래프 데이터베이스"""
    
    def __init__(self, use_neo4j: bool = True):
        self.use_neo4j = use_neo4j and HAS_NEO4J
        self.driver = None
        self.graph = None
        self._lock = threading.Lock()
        
        if self.use_neo4j:
            self._initialize_neo4j()
        else:
            self._initialize_networkx()
    
    def _initialize_neo4j(self):
        """Neo4j 초기화"""
        try:
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            user = os.getenv('NEO4J_USER', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'opencodeai')
            
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.success("Neo4j 그래프 DB 연결 완료")
            
        except Exception as e:
            logger.warning(f"Neo4j 연결 실패, NetworkX로 전환: {e}")
            self.use_neo4j = False
            self._initialize_networkx()
    
    def _initialize_networkx(self):
        """NetworkX 초기화"""
        try:
            self.graph = nx.MultiDiGraph()
            
            # 저장된 그래프 로드 시도
            graph_file = f"{settings.GRAPH_DB_PATH}/graph.gml"
            if os.path.exists(graph_file):
                self.graph = nx.read_gml(graph_file)
                logger.info(f"기존 그래프 로드: {self.graph.number_of_nodes()} 노드, {self.graph.number_of_edges()} 간선")
            
            logger.success("NetworkX 그래프 DB 초기화 완료")
            
        except Exception as e:
            logger.error(f"NetworkX 초기화 실패: {e}")
            self.graph = nx.MultiDiGraph()
    
    def add_file_node(self, file_path: str, metadata: Dict[str, Any]):
        """파일 노드 추가"""
        try:
            with self._lock:
                if self.use_neo4j:
                    self._add_neo4j_file_node(file_path, metadata)
                else:
                    self._add_networkx_file_node(file_path, metadata)
                    
        except Exception as e:
            logger.error(f"파일 노드 추가 실패: {e}")
    
    def _add_neo4j_file_node(self, file_path: str, metadata: Dict[str, Any]):
        """Neo4j에 파일 노드 추가"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (f:File {path: $path})
                SET f.name = $name,
                    f.extension = $extension,
                    f.size = $size,
                    f.last_modified = $last_modified,
                    f.language = $language
                """,
                path=file_path,
                name=os.path.basename(file_path),
                extension=os.path.splitext(file_path)[1],
                size=metadata.get('size', 0),
                last_modified=metadata.get('last_modified', ''),
                language=metadata.get('language', 'unknown')
            )
    
    def _add_networkx_file_node(self, file_path: str, metadata: Dict[str, Any]):
        """NetworkX에 파일 노드 추가"""
        node_attrs = {
            'type': 'file',
            'name': os.path.basename(file_path),
            'extension': os.path.splitext(file_path)[1],
            **metadata
        }
        self.graph.add_node(file_path, **node_attrs)
    
    def add_function_node(self, file_path: str, function_name: str, metadata: Dict[str, Any]):
        """함수 노드 추가 및 파일과 연결"""
        try:
            function_id = f"{file_path}::{function_name}"
            
            with self._lock:
                if self.use_neo4j:
                    self._add_neo4j_function_node(file_path, function_id, function_name, metadata)
                else:
                    self._add_networkx_function_node(file_path, function_id, function_name, metadata)
                    
        except Exception as e:
            logger.error(f"함수 노드 추가 실패: {e}")
    
    def _add_neo4j_function_node(self, file_path: str, function_id: str, function_name: str, metadata: Dict[str, Any]):
        """Neo4j에 함수 노드 추가"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (f:Function {id: $function_id})
                SET f.name = $name,
                    f.start_line = $start_line,
                    f.end_line = $end_line,
                    f.complexity = $complexity,
                    f.parameters = $parameters
                
                WITH f
                MATCH (file:File {path: $file_path})
                MERGE (file)-[:CONTAINS]->(f)
                """,
                function_id=function_id,
                name=function_name,
                start_line=metadata.get('start_line', 0),
                end_line=metadata.get('end_line', 0),
                complexity=metadata.get('complexity', 1),
                parameters=json.dumps(metadata.get('parameters', [])),
                file_path=file_path
            )
    
    def _add_networkx_function_node(self, file_path: str, function_id: str, function_name: str, metadata: Dict[str, Any]):
        """NetworkX에 함수 노드 추가"""
        func_attrs = {
            'type': 'function',
            'name': function_name,
            **metadata
        }
        self.graph.add_node(function_id, **func_attrs)
        self.graph.add_edge(file_path, function_id, relationship='contains')
    
    def add_dependency(self, from_file: str, to_file: str, import_type: str = 'import'):
        """파일 간 의존성 관계 추가"""
        try:
            with self._lock:
                if self.use_neo4j:
                    self._add_neo4j_dependency(from_file, to_file, import_type)
                else:
                    self._add_networkx_dependency(from_file, to_file, import_type)
                    
        except Exception as e:
            logger.error(f"의존성 추가 실패: {e}")
    
    def _add_neo4j_dependency(self, from_file: str, to_file: str, import_type: str):
        """Neo4j에 의존성 관계 추가"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (from:File {path: $from_file})
                MATCH (to:File {path: $to_file})
                MERGE (from)-[r:DEPENDS_ON {type: $import_type}]->(to)
                """,
                from_file=from_file,
                to_file=to_file,
                import_type=import_type
            )
    
    def _add_networkx_dependency(self, from_file: str, to_file: str, import_type: str):
        """NetworkX에 의존성 관계 추가"""
        self.graph.add_edge(from_file, to_file, relationship='depends_on', type=import_type)
    
    def find_related_files(self, file_path: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """관련 파일 찾기
        
        Args:
            file_path: 기준 파일 경로
            max_depth: 탐색 깊이 (기본값: 2)
            
        Returns:
            관련 파일 정보 리스트
        """
        if not self.graph.has_node(file_path):
            return []
        
        visited = set()
        related_files = []
        
        def dfs(node: str, depth: int, weight: float = 1.0):
            """깊이 우선 탐색으로 관련 파일 찾기
            
            Args:
                node: 현재 노드
                depth: 현재 깊이
                weight: 관계 가중치 (깊이에 따라 감소)
            """
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            # 현재 노드의 이웃 탐색
            for neighbor in self.graph.neighbors(node):
                if neighbor == file_path:
                    continue
                
                # 파일 노드인 경우 결과에 추가
                if self.graph.nodes[neighbor].get('type') == 'file':
                    # 관계 정보 수집
                    edge_data = self.graph.get_edge_data(node, neighbor)
                    relationship = edge_data.get('relationship', 'unknown') if edge_data else 'unknown'
                    
                    # 가중치 계산 (깊이와 관계 유형에 따라)
                    current_weight = weight * (0.8 ** depth)  # 깊이에 따라 가중치 감소
                    if relationship in ['imports', 'depends_on']:
                        current_weight *= 1.2  # 직접적인 의존성은 가중치 증가
                    
                    related_files.append({
                        'file_path': neighbor,
                        'relationship': relationship,
                        'depth': depth,
                        'weight': current_weight,
                        'metadata': self.graph.nodes[neighbor].get('metadata', {})
                    })
                
                # 다음 깊이 탐색
                if depth < max_depth:
                    dfs(neighbor, depth + 1, weight * 0.8)
                
        # 탐색 시작
        dfs(file_path, 0)
        
        # 가중치 기준 정렬 및 중복 제거
        seen = set()
        unique_files = []
        for file_info in sorted(related_files, key=lambda x: (-x['weight'], x['depth'])):
            if file_info['file_path'] not in seen:
                seen.add(file_info['file_path'])
                unique_files.append(file_info)
        
        # 최대 20개 반환
        return unique_files[:20]
    
    def save_graph(self):
        """그래프 저장"""
        try:
            if not self.use_neo4j and self.graph:
                os.makedirs(settings.GRAPH_DB_PATH, exist_ok=True)
                graph_file = f"{settings.GRAPH_DB_PATH}/graph.gml"
                nx.write_gml(self.graph, graph_file)
                logger.info("NetworkX 그래프 저장 완료")
                
        except Exception as e:
            logger.error(f"그래프 저장 실패: {e}")
    
    def close(self):
        """연결 종료"""
        if self.use_neo4j and self.driver:
            self.driver.close()
        elif self.graph:
            self.save_graph()

    def add_call_relation(self, caller_id: str, callee_id: str):
        """함수 간 CALLS 엣지 추가 (caller_id, callee_id는 'file_path::func_name' 형식)"""
        try:
            with self._lock:
                if self.use_neo4j:
                    with self.driver.session() as session:
                        session.run(
                            """
                            MERGE (caller:Function {id: $caller_id})
                            MERGE (callee:Function {id: $callee_id})
                            MERGE (caller)-[:CALLS]->(callee)
                            """,
                            caller_id=caller_id, callee_id=callee_id
                        )
                else:
                    self.graph.add_edge(caller_id, callee_id, relationship='calls')
        except Exception as e:
            logger.error(f"CALLS 엣지 추가 실패: {e}")

    def search_file_by_name(self, filename: str) -> list:
        """그래프 DB에서 파일명으로 전체 경로 후보 반환"""
        if hasattr(self, 'graph') and self.graph:
            return [n for n in self.graph.nodes if os.path.basename(n) == filename or filename in os.path.basename(n)]
        return []

    def get_key_nodes(self, top_n: int = 10) -> list:
        """
        프로젝트 전체 분석용 주요 노드(진입점, 허브, 연결 많은 함수/파일 등) 추출
        """
        if not hasattr(self, 'graph') or self.graph is None:
            return []
        # 파일/함수 노드 중 degree(연결 수) 기준 상위 top_n 추출
        nodes = [(n, self.graph.degree(n)) for n in self.graph.nodes if self.graph.nodes[n].get('type') in {'file', 'function'}]
        nodes_sorted = sorted(nodes, key=lambda x: x[1], reverse=True)
        return [n for n, _ in nodes_sorted[:top_n]]

class CodeIndexer:
    """코드 인덱싱 시스템"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.graph_db = GraphDatabase()
        self.code_analyzer = get_code_analyzer()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.metadata_db_path = settings.METADATA_DB_PATH
        self._initialize_metadata_db()
        # 인덱싱 상태
        self.indexing_in_progress = False
        self.indexing_progress = 0
        self.indexing_total = 0
        self.indexing_error = None
    
    def _initialize_metadata_db(self):
        """메타데이터 DB 초기화"""
        try:
            os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    last_indexed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chunk_count INTEGER DEFAULT 0,
                    language TEXT,
                    file_size INTEGER
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indexing_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_files INTEGER,
                    total_chunks INTEGER,
                    last_full_index TIMESTAMP,
                    index_version TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("메타데이터 DB 초기화 완료")
            
        except Exception as e:
            logger.error(f"메타데이터 DB 초기화 실패: {e}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def _should_reindex_file(self, file_path: str) -> bool:
        """파일 재인덱싱 필요 여부 확인 (벡터 DB에도 청크가 실제로 존재하는지까지 확인)"""
        try:
            # 1. 벡터 DB에 해당 파일의 청크가 하나도 없으면 반드시 재인덱싱
            if file_path not in self.vector_db.file_index or not self.vector_db.file_index[file_path]:
                logger.warning(f"[REINDEX] 벡터 DB에 청크 없음: {file_path}")
                return True
            # 2. 메타데이터 DB 체크는 그 다음
            current_hash = self._get_file_hash(file_path)
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT file_hash FROM indexed_files WHERE file_path = ?", (file_path,))
            result = cursor.fetchone()
            conn.close()
            if result is None:
                return True  # 새 파일
            return result[0] != current_hash  # 해시가 다르면 재인덱싱
        except Exception as e:
            logger.error(f"파일 해시 비교 실패: {e}")
            return True
    
    async def index_file(self, file_path: str) -> bool:
        logger.info(f"[DEBUG] index_file 진입: {file_path} | vector_db.file_index keys={list(self.vector_db.file_index.keys())[:3]} 전체길이={len(self.vector_db.file_index)}")
        try:
            if not os.path.exists(file_path):
                logger.warning(f"파일이 존재하지 않습니다: {file_path}")
                return False
            # 재인덱싱 필요 여부 확인
            if not self._should_reindex_file(file_path):
                logger.debug(f"파일이 이미 최신 상태입니다: {file_path}")
                return True
            logger.info(f"[INDEX] 파일 인덱싱 시작: {file_path}")
            # 기존 데이터 제거
            self.vector_db.remove_file(file_path)
            # 코드 분석
            logger.error(f"[DEBUG] analyze_file 호출 직전: {file_path}")
            analysis_result = await self.code_analyzer.analyze_file(file_path)
            logger.error(f"[DEBUG] analyze_file 호출 후: {file_path} 결과 keys={list(analysis_result.keys()) if analysis_result else 'None'}")
            if not analysis_result:
                logger.warning(f"[INDEX] 코드 분석 실패: {file_path}")
                return False
            logger.info(f"[INDEX] 코드 분석 결과: {file_path} - 함수 {len(analysis_result.get('functions', []))}개, 클래스 {len(analysis_result.get('classes', []))}개")
            # 파일을 청크로 분할
            chunks = await self._chunk_file(file_path, analysis_result)
            logger.info(f"[INDEX] 청크 생성 결과: {file_path} - 청크 {len(chunks)}개")
            if not chunks:
                logger.warning(f"[INDEX] 청크가 하나도 생성되지 않음: {file_path}")
            # 벡터 DB에 추가
            if chunks:
                logger.error(f"[DEBUG] add_chunks 호출 직전: {file_path} chunks={len(chunks)}")
                await self.vector_db.add_chunks(chunks)
                logger.error(f"[DEBUG] add_chunks 호출 후: {file_path}")
            # 그래프 DB에 추가
            await self._add_to_graph(file_path, analysis_result)
            # 메타데이터 업데이트
            self._update_file_metadata(file_path, len(chunks), analysis_result['language'])
            logger.success(f"[INDEX] 파일 인덱싱 완료: {file_path} ({len(chunks)} 청크)")
            return True
        except Exception as e:
            logger.error(f"[INDEX] 파일 인덱싱 실패 {file_path}: {e}\n{traceback.format_exc()}")
            return False
    
    async def _chunk_file(self, file_path: str, analysis: Dict[str, Any]) -> List[CodeChunk]:
        chunks = []
        logger.debug(f"[CHUNK_FILE] 진입: file_path={file_path}, 분석 keys={list(analysis.keys())}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            lines = content.split('\n')
            # 함수 기반 청크 생성
            if analysis.get('functions'):
                logger.debug(f"[CHUNK_FILE] 함수 청크 {len(analysis['functions'])}개 생성 시도")
                for func in analysis['functions']:
                    chunk_content = '\n'.join(lines[func['start_line']-1:func['end_line']])
                    chunk_id = f"{file_path}::func::{func['name']}"
                    
                    chunk = CodeChunk(
                        id=chunk_id,
                        file_path=file_path,
                        content=chunk_content,
                        start_line=func['start_line'],
                        end_line=func['end_line'],
                        language=analysis['language'],
                        chunk_type='function',
                        metadata={
                            'function_name': func['name'],
                            'parameters': func['parameters'],
                            'complexity': func['complexity'],
                            'docstring': func['docstring']
                        }
                    )
                    chunks.append(chunk)
            # 클래스 기반 청크 생성
            if analysis.get('classes'):
                logger.debug(f"[CHUNK_FILE] 클래스 청크 {len(analysis['classes'])}개 생성 시도")
                for cls in analysis['classes']:
                    chunk_content = '\n'.join(lines[cls['start_line']-1:cls['end_line']])
                    chunk_id = f"{file_path}::class::{cls['name']}"
                    
                    chunk = CodeChunk(
                        id=chunk_id,
                        file_path=file_path,
                        content=chunk_content,
                        start_line=cls['start_line'],
                        end_line=cls['end_line'],
                        language=analysis['language'],
                        chunk_type='class',
                        metadata={
                            'class_name': cls['name'],
                            'methods': [m['name'] for m in cls['methods']],
                            'inheritance': cls['inheritance'],
                            'docstring': cls['docstring']
                        }
                    )
                    chunks.append(chunk)
            # 고정 크기 청크 생성 (함수/클래스가 없는 부분)
            chunk_size = 1000
            overlap = 200
            
            # 이미 처리된 라인들 추적
            processed_lines = set()
            for chunk in chunks:
                for line_no in range(chunk.start_line, chunk.end_line + 1):
                    processed_lines.add(line_no)
            
            # 처리되지 않은 라인들을 고정 크기 청크로 분할
            current_chunk_lines = []
            current_start_line = 1
            
            for i, line in enumerate(lines, 1):
                if i not in processed_lines:
                    if not current_chunk_lines:
                        current_start_line = i
                    
                    current_chunk_lines.append(line)
                    
                    if len('\n'.join(current_chunk_lines)) >= chunk_size:
                        # 청크 생성
                        chunk_content = '\n'.join(current_chunk_lines)
                        chunk_id = f"{file_path}::chunk::{current_start_line}:{i}"
                        
                        chunk = CodeChunk(
                            id=chunk_id,
                            file_path=file_path,
                            content=chunk_content,
                            start_line=current_start_line,
                            end_line=i,
                            language=analysis['language'],
                            chunk_type='other',
                            metadata={'chunk_type': 'fixed_size'}
                        )
                        chunks.append(chunk)
                        
                        # 오버랩 처리
                        if overlap > 0 and len(current_chunk_lines) > overlap:
                            current_chunk_lines = current_chunk_lines[-overlap:]
                            current_start_line = i - overlap + 1
                        else:
                            current_chunk_lines = []
                else:
                    # 처리된 라인을 만나면 현재 청크 마무리
                    if current_chunk_lines:
                        chunk_content = '\n'.join(current_chunk_lines)
                        chunk_id = f"{file_path}::chunk::{current_start_line}:{i-1}"
                        
                        chunk = CodeChunk(
                            id=chunk_id,
                            file_path=file_path,
                            content=chunk_content,
                            start_line=current_start_line,
                            end_line=i-1,
                            language=analysis['language'],
                            chunk_type='other',
                            metadata={'chunk_type': 'fixed_size'}
                        )
                        chunks.append(chunk)
                        current_chunk_lines = []
            
            # 마지막 청크 처리
            if current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines)
                chunk_id = f"{file_path}::chunk::{current_start_line}:{len(lines)}"
                
                chunk = CodeChunk(
                    id=chunk_id,
                    file_path=file_path,
                    content=chunk_content,
                    start_line=current_start_line,
                    end_line=len(lines),
                    language=analysis['language'],
                    chunk_type='other',
                    metadata={'chunk_type': 'fixed_size'}
                )
                chunks.append(chunk)
            
            logger.debug(f"[CHUNK_FILE] 최종 생성된 청크 개수: {len(chunks)} (file_path={file_path})")
        except Exception as e:
            logger.error(f"[CHUNK_FILE] 파일 청킹 실패 {file_path}: {e}")
        return chunks
    
    async def _add_to_graph(self, file_path: str, analysis: Dict[str, Any]):
        """그래프 DB에 분석 결과 추가 (내부 import, 함수 호출 관계 포함)"""
        try:
            file_metadata = {
                'language': analysis['language'],
                'lines_of_code': analysis['lines_of_code'],
                'complexity': analysis['complexity'],
                'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            self.graph_db.add_file_node(file_path, file_metadata)

            func_name_to_id = {}
            if analysis.get('functions'):
                for func in analysis['functions']:
                    func_metadata = {
                        'start_line': func['start_line'],
                        'end_line': func['end_line'],
                        'complexity': func['complexity'],
                        'parameters': func['parameters'],
                        'is_async': func.get('is_async', False)
                    }
                    self.graph_db.add_function_node(file_path, func['name'], func_metadata)
                    func_name_to_id[func['name']] = f"{file_path}::{func['name']}"

            if analysis.get('imports'):
                for imp in analysis['imports']:
                    if imp['module']:
                        if imp['module'].startswith('.') or imp['module'].replace('.', '/') in file_path:
                            base_dir = os.path.dirname(file_path)
                            rel_path = imp['module'].replace('.', '/') + '.py'
                            internal_path = os.path.normpath(os.path.join(base_dir, rel_path))
                            if os.path.exists(internal_path):
                                self.graph_db.add_dependency(file_path, internal_path, 'import')
                        else:
                            dependency_path = f"module::{imp['module']}"
                            self.graph_db.add_dependency(file_path, dependency_path, 'import')

            if analysis.get('functions'):
                for func in analysis['functions']:
                    caller_id = f"{file_path}::{func['name']}"
                    if func.get('calls'):
                        for callee_name in func['calls']:
                            if callee_name in func_name_to_id:
                                callee_id = func_name_to_id[callee_name]
                                self.graph_db.add_call_relation(caller_id, callee_id)

            # Java 매퍼 연동: mappers 필드가 있으면 그래프에 추가
            if analysis.get('mappers'):
                mapper_file = None
                dir_path = os.path.dirname(file_path)
                for fname in os.listdir(dir_path):
                    if fname.endswith('Mapper.xml'):
                        mapper_file = os.path.join(dir_path, fname)
                        break
                if mapper_file:
                    self.graph_db.add_file_node(mapper_file, {'type': 'xml_mapper'})
                    for entry in analysis['mappers']:
                        query_id = f"{mapper_file}::{entry['id']}"
                        self.graph_db.add_function_node(mapper_file, entry['id'], {
                            'type': entry['type'],
                            'parameterType': entry['parameterType'],
                            'resultType': entry['resultType'],
                            'sql': entry['sql']
                        })
                        # Java 함수와 쿼리 id가 일치하면 연결
                        if entry['id'] in func_name_to_id:
                            self.graph_db.add_call_relation(func_name_to_id[entry['id']], query_id)
        except Exception as e:
            logger.error(f"그래프 DB 추가 실패 {file_path}: {e}")
    
    def _update_file_metadata(self, file_path: str, chunk_count: int, language: str):
        """파일 메타데이터 업데이트"""
        try:
            file_hash = self._get_file_hash(file_path)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO indexed_files 
                (file_path, file_hash, chunk_count, language, file_size, last_indexed)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (file_path, file_hash, chunk_count, language, file_size))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"메타데이터 업데이트 실패: {e}")
    
    async def index_directory(self, directory_path: str = None, max_files: int = None) -> Dict[str, Any]:
        # 모든 입력 경로를 무조건 /workspace로 고정
        directory_path = "/workspace"
        logger.error(f"[DEBUG] index_directory 진입: vector_db.file_index keys={list(self.vector_db.file_index.keys())[:3]} 전체길이={len(self.vector_db.file_index)}, chunk_map keys={list(self.vector_db.chunk_map.keys())[:3]} 전체길이={len(self.vector_db.chunk_map)}")
        try:
            self.indexing_in_progress = True
            self.indexing_progress = 0
            self.indexing_error = None
            directory_path = Path(directory_path)
            if not directory_path.exists():
                raise ValueError(f"디렉토리가 존재하지 않습니다: {directory_path}")
            supported_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']
            if hasattr(settings, 'SUPPORTED_EXTENSIONS') and settings.SUPPORTED_EXTENSIONS:
                supported_extensions = [ext.strip() for ext in settings.SUPPORTED_EXTENSIONS.split(',') if ext.strip()]
            files = []
            for ext in supported_extensions:
                pattern = f"**/*{ext}"
                found_files = list(directory_path.glob(pattern))
                files.extend(found_files)
            # --- .gitignore 및 숨김 디렉토리(.으로 시작) 예외 처리 ---
            exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'dist', 'build'}
            # .gitignore에서 제외할 경로 읽기
            gitignore_path = directory_path / '.gitignore'
            gitignore_patterns = set()
            if gitignore_path.exists():
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # 디렉토리 패턴만 추출
                            if line.endswith('/'):
                                gitignore_patterns.add(line.rstrip('/'))
                            else:
                                gitignore_patterns.add(line)
            def is_excluded(f):
                # .으로 시작하는 디렉토리 예외 (루트/중간 모두)
                if any(part.startswith('.') and part not in {'.', '..'} for part in f.parts):
                    return True
                # 빌드/캐시 디렉토리 예외
                if any(part in exclude_dirs for part in f.parts):
                    return True
                # .gitignore 패턴 예외 (prefix 매칭)
                for pattern in gitignore_patterns:
                    # 디렉토리 패턴이면 prefix 매칭
                    if (str(f).startswith(str(directory_path / pattern)) or
                        (pattern in str(f))):
                        return True
                return False
            files = [f for f in files if not is_excluded(f)]
            if max_files:
                files = files[:max_files]
            self.indexing_total = len(files)
            logger.info(f"[DEBUG] 인덱싱 대상 파일 수: {len(files)} (예시: {[str(f) for f in files[:5]]})")
            logger.info(f"디렉토리 인덱싱 시작: {directory_path} ({len(files)} 파일)")

            # --- rename 감지 및 처리 ---
            import sqlite3
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, file_hash FROM indexed_files")
            old_hash_map = {row[1]: row[0] for row in cursor.fetchall()}
            conn.close()
            for file_path in files:
                file_hash = self._get_file_hash(str(file_path))
                if file_hash in old_hash_map and str(file_path) != old_hash_map[file_hash]:
                    old_path = old_hash_map[file_hash]
                    logger.info(f"[RENAME 감지] {old_path} → {file_path}")
                    # 벡터 DB/그래프 DB/메타데이터에서 기존 경로 삭제
                    self.vector_db.remove_file(old_path)
                    self.graph_db.add_file_node(old_path, {"deleted": True})
                    # 메타데이터 DB에서 기존 경로 삭제
                    conn = sqlite3.connect(self.metadata_db_path)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM indexed_files WHERE file_path = ?", (old_path,))
                    conn.commit()
                    conn.close()
            # --- 기존 인덱싱 루프 ... (아래 기존 코드 유지) ---

            success_count = 0
            error_count = 0
            failed_files = []
            tasks = []
            semaphore = asyncio.Semaphore(8)
            async def index_with_semaphore(file_path):
                async with semaphore:
                    try:
                        return await self.index_file(str(file_path))
                    except Exception as e:
                        failed_files.append(file_path)
                        with open('logs/failed_indexing.log', 'a') as f:
                            f.write(f"{file_path}: {str(e)}\n")
                            return False
            for idx, file_path in enumerate(files):
                task = asyncio.create_task(index_with_semaphore(file_path))
                tasks.append(task)
                logger.info(f"[인덱싱 진행률] {idx+1}/{len(files)} ({(idx+1)/len(files)*100:.1f}%) - {file_path}")
            results = []
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)
                self.indexing_progress = i + 1
                if result:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"파일 인덱싱 실패: {files[i]}")
                if (i + 1) % 10 == 0:
                    logger.info(f"진행상황: {i + 1}/{len(files)} 파일 처리됨")
            self.vector_db._save_index()
            self.graph_db.save_graph()
            self._update_indexing_stats(len(files), success_count)
            self.indexing_in_progress = False
            self.indexing_progress = self.indexing_total
            result = {
                'total_files': len(files),
                'success_count': success_count,
                'error_count': error_count,
                'directory': str(directory_path),
                'timestamp': datetime.now().isoformat(),
                'failed_files': failed_files,
            }
            logger.success(f"디렉토리 인덱싱 완료: {success_count}/{len(files)} 성공")
            return result
        except Exception as e:
            self.indexing_in_progress = False
            self.indexing_error = str(e)
            logger.error(f"디렉토리 인덱싱 실패: {e}\n{traceback.format_exc()}")
            return {'error': str(e)}
    
    def _update_indexing_stats(self, total_files: int, success_count: int):
        """인덱싱 통계 업데이트"""
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # 총 청크 수 계산
            cursor.execute("SELECT SUM(chunk_count) FROM indexed_files")
            total_chunks = cursor.fetchone()[0] or 0
            
            cursor.execute("""
                INSERT INTO indexing_stats (total_files, total_chunks, last_full_index, index_version)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """, (success_count, total_chunks, settings.VERSION))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"통계 업데이트 실패: {e}")
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """인덱싱 통계 조회"""
        try:
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            
            # 파일 통계
            cursor.execute("SELECT COUNT(*), SUM(chunk_count) FROM indexed_files")
            file_stats = cursor.fetchone()
            
            # 언어별 통계
            cursor.execute("""
                SELECT language, COUNT(*), SUM(chunk_count) 
                FROM indexed_files 
                GROUP BY language 
                ORDER BY COUNT(*) DESC
            """)
            language_stats = cursor.fetchall()
            
            # 최근 인덱싱 정보
            cursor.execute("""
                SELECT last_full_index, index_version 
                FROM indexing_stats 
                ORDER BY id DESC 
                LIMIT 1
            """)
            last_index = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_files': file_stats[0] or 0,
                'total_chunks': file_stats[1] or 0,
                'languages': [
                    {'language': lang, 'files': count, 'chunks': chunks}
                    for lang, count, chunks in language_stats
                ],
                'last_full_index': last_index[0] if last_index else None,
                'index_version': last_index[1] if last_index else None,
                'vector_db_size': len(self.vector_db.chunk_map),
                'graph_db_nodes': self.graph_db.graph.number_of_nodes() if hasattr(self.graph_db, 'graph') else 0
            }
            
        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}

def extract_user_query(query):
    """
    query가 dict(messages=...) 또는 str(prompt)일 때, 가장 마지막 user 메시지 content만 추출
    """
    if isinstance(query, dict) and 'messages' in query:
        user_messages = [m['content'] for m in query['messages'] if m.get('role') == 'user']
        return user_messages[-1] if user_messages else ''
    elif isinstance(query, list):
        user_messages = [m['content'] for m in query if m.get('role') == 'user']
        return user_messages[-1] if user_messages else ''
    elif isinstance(query, str):
        # 프롬프트에서 마지막 User: ... 를 추출
        import re
        matches = re.findall(r'User: (.*?)(?:\n|$)', query, re.DOTALL)
        return matches[-1].strip() if matches else query.strip()
    return str(query)

def is_full_project_query(user_query: str) -> bool:
    keywords = [
        '전체 코드', '전체 구조', '전체 파일', '프로젝트 전체', 'all files', 'project overview', '전체 분석', '전체 함수', '전체 클래스', '전체 모듈', '전체 연관관계', '전체 dependency', '전체 의존성'
    ]
    return any(k in user_query.lower() for k in keywords)

class RAGSystem:
    """RAG (Retrieval-Augmented Generation) 시스템"""
    
    def __init__(self):
        self.indexer = CodeIndexer()
        self.vector_db = self.indexer.vector_db
        self.graph_db = self.indexer.graph_db
        self.llm_manager = get_llm_manager()
    
    async def search_codebase(
        self,
        query: str,
        project_path: Optional[str] = None,
        k: int = 50,
        include_graph: bool = True
    ) -> list:
        logger.info(f"[RAGSystem] search_codebase 진입: query='{str(query)[:50]}...', k={k}, include_graph={include_graph}")
        try:
            user_query = extract_user_query(query)
            logger.info(f"코드베이스 검색: {user_query[:50]}...")
            if not self.vector_db.chunk_map or len(self.vector_db.chunk_map) == 0:
                logger.warning("[RAGSystem] 벡터 DB가 비어 있음. 자동 재인덱싱 시도...")
                await self.indexer.index_directory("/workspace")
            if user_query.strip().endswith('.js') or user_query.strip().endswith('.py') or user_query.strip().endswith('.ts'):
                matches = self.vector_db.search_file_by_name(user_query.strip())
                if matches:
                    results = []
                    for path in matches:
                        chunk_ids = self.vector_db.file_index.get(path, [])
                        if chunk_ids:
                            chunk = self.vector_db.chunk_map[chunk_ids[0]]
                            sr = SearchResult(chunk=chunk, similarity_score=1.0)
                            results.append(sr)
                    logger.info(f"[RAGSystem] 파일명 기반 검색 결과: {len(results)}개")
                    return results[:k]
            vector_results = await self.vector_db.search(user_query, k)
            if include_graph:
                for result in vector_results:
                    file_path = result.chunk.file_path
                    related_files = self.graph_db.find_related_files(file_path)
                    result.graph_connections = related_files
                    graph_boost = len(related_files) * 0.1
                    result.relevance_score = result.similarity_score + graph_boost
            vector_results.sort(key=lambda x: x.relevance_score or x.similarity_score, reverse=True)
            logger.info(f"[RAGSystem] search_codebase 결과: {len(vector_results)}개")
            return vector_results
        except Exception as e:
            logger.error(f"코드베이스 검색 실패: {e}")
            return []
    
    async def get_context_for_query(
        self, 
        query: str, 
        max_context_length: int = 4000
    ) -> str:
        logger.debug(f"[RAGSystem] get_context_for_query 진입: query='{str(query)[:50]}...'")
        try:
            user_query = extract_user_query(query)
            # 전체 프로젝트/구조/연관관계 요청이면 분할 분석
            if is_full_project_query(user_query):
                # 1. graph DB에서 주요 노드 추출
                key_nodes = self.graph_db.get_key_nodes(top_n=10)
                logger.info(f"[RAGSystem] 전체 분석용 주요 노드: {key_nodes}")
                # 2. 각 노드별로 vector DB에서 chunk 추출
                partial_contexts = []
                for node in key_nodes:
                    # 파일 노드면 해당 파일의 첫 chunk, 함수 노드면 id 매칭 chunk
                    chunk = None
                    if node in self.vector_db.file_index:
                        chunk_ids = self.vector_db.file_index[node]
                        if chunk_ids:
                            chunk = self.vector_db.chunk_map[chunk_ids[0]]
                    elif node in self.vector_db.chunk_map:
                        chunk = self.vector_db.chunk_map[node]
                    if chunk:
                        partial_contexts.append(chunk)
                # 3. 각 chunk별로 LLM에 부분 분석 요청 (비동기)
                async def analyze_chunk(chunk):
                    prompt = f"""아래 코드의 역할과 다른 파일/함수와의 연관관계를 요약해줘.\n\n# 파일: {chunk.file_path}\n# 라인: {chunk.start_line}-{chunk.end_line}\n# 타입: {chunk.chunk_type}\n\n```{chunk.language}\n{chunk.content}\n```"""
                    return await self.llm_manager.generate_response(prompt, max_tokens=512, temperature=0.1)
                import asyncio
                analyses = await asyncio.gather(*[analyze_chunk(chunk) for chunk in partial_contexts])
                # 4. 부분 분석 결과를 LLM에 통합 요약 요청
                summary_prompt = f"""아래는 프로젝트 주요 파일/함수별 분석 결과입니다.\n각 부분을 종합해 전체 코드 구조, 주요 기능, 연관관계를 한눈에 볼 수 있게 요약해줘.\n\n"""
                for i, (chunk, analysis) in enumerate(zip(partial_contexts, analyses)):
                    summary_prompt += f"## [{i+1}] 파일/함수: {chunk.file_path} ({chunk.chunk_type})\n분석: {analysis}\n\n"
                summary_prompt += "\n최종 요약:"
                final_summary = await self.llm_manager.generate_response(summary_prompt, max_tokens=1024, temperature=0.1)
                return final_summary
            # 기존 방식 (부분 검색)
            context_parts = []
            current_length = 0
            k = 5
            search_results = await self.search_codebase(user_query, k=k)
            if not search_results:
                return ""
            for result in search_results:
                chunk = result.chunk
                context_part = f"""
# File: {chunk.file_path} (Lines {chunk.start_line}-{chunk.end_line})
# Type: {chunk.chunk_type}
# Similarity: {result.similarity_score:.3f}

```{chunk.language}
{chunk.content}
```
"""
                if current_length + len(context_part) > max_context_length:
                    break
                context_parts.append(context_part)
                current_length += len(context_part)
            logger.info(f"[RAGSystem] 컨텍스트 생성 완료: {len(context_parts)}개 파트, 총 길이 {current_length}")
            return "\n".join(context_parts)
        except Exception as e:
            logger.error(f"컨텍스트 생성 실패: {e}")
            return ""
    
    async def answer_with_context(self, query: str, context: str = None) -> str:
        """컨텍스트를 활용한 답변 생성"""
        try:
            if context is None:
                context = await self.get_context_for_query(query)
            
            # 프롬프트 구성
            prompt = f"""당신은 전문적인 소프트웨어 개발자입니다. 주어진 코드 컨텍스트를 바탕으로 질문에 답변하세요.

## 코드 컨텍스트:
{context}

## 질문:
{query}

## 답변:
위의 코드 컨텍스트를 참고하여 정확하고 구체적인 답변을 제공하세요. 필요시 코드 예시도 포함하세요."""

            # LLM으로 답변 생성
            response = await self.llm_manager.generate_response(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.1
            )
            
            return response
            
        except Exception as e:
            logger.error(f"컨텍스트 답변 생성 실패: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다."

# 전역 RAG 시스템 인스턴스
_rag_system_instance = None

def get_rag_system() -> RAGSystem:
    """전역 RAG 시스템 인스턴스 반환"""
    global _rag_system_instance
    
    if _rag_system_instance is None:
        _rag_system_instance = RAGSystem()
    
    return _rag_system_instance