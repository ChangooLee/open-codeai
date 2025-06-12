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
        self.dimension = int(os.getenv("EMBEDDING_MODEL_DIM", 768)) if dimension is None else dimension
        self.index_path: str = index_path or settings.VECTOR_INDEX_PATH
        self.index = None
        self.chunk_map: dict[str, CodeChunk] = {}
        self.file_index: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """인덱스 초기화"""
        if not HAS_FAISS:
            raise ImportError("FAISS is required for vector database")
            
        try:
            # HNSW 인덱스 사용 (대용량 데이터에 적합)
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
            
            # 기존 인덱스 로드 시도
            if os.path.exists(f"{self.index_path}/vector.index"):
                self._load_index()
                
            logger.info(f"벡터 DB 초기화 완료 - 차원: {self.dimension}")
                
        except Exception as e:
            logger.error(f"벡터 DB 초기화 실패: {e}")
            raise
    
    def _load_index(self) -> None:
        """저장된 인덱스 로드"""
        try:
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
            raise
    
    def _save_index(self) -> None:
        """인덱스 저장"""
        try:
            if not self.index:
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
            raise
    
    async def add_chunks(self, chunks: List[CodeChunk]) -> bool:
        """청크를 벡터 DB에 추가"""
        if not chunks:
            return False
        
        try:
            # 첫 번째 청크의 임베딩 차원 검증
            first_chunk = chunks[0]
            if not first_chunk.embedding:
                raise ValueError("임베딩이 없는 청크가 있습니다")
            
            actual_dim = len(first_chunk.embedding)
            if actual_dim != self.dimension:
                raise ValueError(f"임베딩 차원 불일치: {actual_dim} != {self.dimension}")
            
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
                raise ValueError("유효한 청크가 없습니다")
            
            # FAISS 인덱스에 추가
            with self._lock:
                self.index.add(np.array(embeddings))
                
                # 청크 매핑 정보 업데이트
                for chunk in valid_chunks:
                    self.chunk_map[chunk.id] = chunk
                    if chunk.file_path not in self.file_index:
                        self.file_index[chunk.file_path] = []
                    self.file_index[chunk.file_path].append(chunk.id)
                
                # 인덱스 저장
                self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"청크 추가 실패: {e}")
            return False
    
    async def search(self, query: str, k: int = 10) -> List[SearchResult]:
        """벡터 검색 수행"""
        try:
            # 쿼리 임베딩 생성
            llm_manager = get_llm_manager()
            query_embedding = await llm_manager.get_embedding(query)
            
            if not query_embedding:
                return []
            
            # FAISS 검색
            with self._lock:
                distances, indices = self.index.search(np.array([query_embedding]), k)
            
            # 결과 변환
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunk_map):
                    chunk_id = list(self.chunk_map.keys())[idx]
                    chunk = self.chunk_map[chunk_id]
                    results.append(SearchResult(
                        chunk=chunk,
                        similarity_score=float(1.0 - distances[0][i])
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            return []

class GraphDatabase:
    """그래프 데이터베이스"""
    
    def __init__(self):
        self.use_neo4j = False
        self.driver = None
        self.graph = nx.DiGraph()
        
        # Neo4j 설정
        self.neo4j_uri = getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = getattr(settings, 'NEO4J_USER', 'neo4j')
        self.neo4j_password = getattr(settings, 'NEO4J_PASSWORD', 'password')
        
        # Neo4j 초기화 시도
        self._initialize_neo4j()
        
        # NetworkX 초기화
        self._initialize_networkx()
        
    def _initialize_neo4j(self) -> None:
        """Neo4j 초기화"""
        if not HAS_NEO4J:
            logger.warning("Neo4j 패키지가 설치되지 않아 NetworkX로 대체됩니다")
            return
            
        try:
            from neo4j import GraphDatabase
            # Docker 환경에서는 neo4j 서비스 이름으로 연결
            uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
            user = os.getenv('NEO4J_USER', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'opencodeai')
            
            logger.info(f"Neo4j 연결 시도: {uri}")
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, password)
            )
            
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")
                
            self.use_neo4j = True
            logger.info("Neo4j 초기화 완료")
            
        except Exception as e:
            logger.error(f"Neo4j 초기화 실패: {e}")
            logger.info("NetworkX로 대체됩니다")
    
    def _initialize_networkx(self):
        """NetworkX 초기화"""
        self.graph = nx.DiGraph()
        logger.info("NetworkX 초기화 완료")
    
    def add_file_node(self, file_path: str, metadata: Dict[str, Any]):
        """파일 노드 추가"""
        with self._lock:
            if self.use_neo4j:
                self._add_neo4j_file_node(file_path, metadata)
            else:
                self._add_networkx_file_node(file_path, metadata)
    
    def _add_neo4j_file_node(self, file_path: str, metadata: Dict[str, Any]):
        """Neo4j에 파일 노드 추가"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (f:File {path: $path})
                SET f += $metadata
                """,
                path=file_path,
                metadata=metadata
            )
    
    def _add_networkx_file_node(self, file_path: str, metadata: Dict[str, Any]):
        """NetworkX에 파일 노드 추가"""
        self.graph.add_node(file_path, type='file', **metadata)
    
    def add_function_node(self, file_path: str, function_name: str, metadata: Dict[str, Any]):
        """함수 노드 추가"""
        with self._lock:
            if self.use_neo4j:
                self._add_neo4j_function_node(file_path, function_name, metadata)
            else:
                self._add_networkx_function_node(file_path, function_name, metadata)
    
    def _add_neo4j_function_node(self, file_path: str, function_name: str, metadata: Dict[str, Any]):
        """Neo4j에 함수 노드 추가"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (f:File {path: $file_path})
                MERGE (func:Function {name: $function_name})
                SET func += $metadata
                MERGE (f)-[:CONTAINS]->(func)
                """,
                file_path=file_path,
                function_name=function_name,
                metadata=metadata
            )
    
    def _add_networkx_function_node(self, file_path: str, function_name: str, metadata: Dict[str, Any]):
        """NetworkX에 함수 노드 추가"""
        function_id = f"{file_path}:{function_name}"
        self.graph.add_node(function_id, type='function', **metadata)
        self.graph.add_edge(file_path, function_id, type='contains')
    
    def add_dependency(self, from_file: str, to_file: str, import_type: str = 'import'):
        """파일 간 의존성 추가"""
        with self._lock:
            if self.use_neo4j:
                self._add_neo4j_dependency(from_file, to_file, import_type)
            else:
                self._add_networkx_dependency(from_file, to_file, import_type)
    
    def _add_neo4j_dependency(self, from_file: str, to_file: str, import_type: str):
        """Neo4j에 의존성 추가"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (from:File {path: $from_file})
                MATCH (to:File {path: $to_file})
                MERGE (from)-[r:IMPORTS {type: $import_type}]->(to)
                """,
                from_file=from_file,
                to_file=to_file,
                import_type=import_type
            )
    
    def _add_networkx_dependency(self, from_file: str, to_file: str, import_type: str):
        """NetworkX에 의존성 추가"""
        self.graph.add_edge(from_file, to_file, type='imports', import_type=import_type)
    
    def find_related_files(self, file_path: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """관련 파일 찾기"""
        with self._lock:
            if self.use_neo4j:
                return self._find_neo4j_related_files(file_path, max_depth)
            else:
                return self._find_networkx_related_files(file_path, max_depth)
    
    def _find_neo4j_related_files(self, file_path: str, max_depth: int) -> List[Dict[str, Any]]:
        """Neo4j에서 관련 파일 찾기"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (f:File {path: $file_path})-[r:IMPORTS*1..$max_depth]->(related:File)
                RETURN related.path as path,
                       length(path) as depth,
                       [rel in relationships(path) | rel.type] as types
                """,
                file_path=file_path,
                max_depth=max_depth
            )
            return [dict(record) for record in result]
    
    def _find_networkx_related_files(self, file_path: str, max_depth: int) -> List[Dict[str, Any]]:
        """NetworkX에서 관련 파일 찾기"""
        if file_path not in self.graph:
            return []
        
        results = []
        visited = set()
        
        def dfs(node: str, depth: int, weight: float = 1.0):
            if depth > max_depth or node in visited:
                return
            
            visited.add(node)
            
            for neighbor in self.graph.successors(node):
                edge = self.graph.get_edge_data(node, neighbor)
                if edge and edge.get('type') == 'imports':
                    results.append({
                        'path': neighbor,
                        'depth': depth,
                        'types': [edge.get('import_type', 'import')],
                        'weight': weight
                    })
                    dfs(neighbor, depth + 1, weight * 0.8)
        
        dfs(file_path, 1)
        return results
    
    def save_graph(self):
        """그래프 저장"""
        if not self.use_neo4j:
            nx.write_gpickle(self.graph, f"{settings.GRAPH_INDEX_PATH}/graph.pickle")
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.use_neo4j and self.driver:
            self.driver.close()

class RAGSystem:
    """RAG 시스템"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.graph_db = GraphDatabase()
        self.code_analyzer = get_code_analyzer()
        self.llm_manager = get_llm_manager()
        self.indexer = CodeIndexer(self)
    
    async def search_codebase(
        self,
        query: str,
        project_path: Optional[str] = None,
        k: int = 50,
        include_graph: bool = True
    ) -> list:
        """코드베이스 검색"""
        try:
            # 벡터 검색
            vector_results = await self.vector_db.search(query, k)
            
            # 그래프 검색
            graph_results = []
            if include_graph:
                for result in vector_results:
                    file_path = result.chunk.file_path
                    related = self.graph_db.find_related_files(file_path)
                    graph_results.extend(related)
            
            # 결과 통합 및 정렬
            combined_results = self._combine_search_results(vector_results, graph_results)
            return self._rank_results(combined_results)
            
        except Exception as e:
            logger.error(f"코드베이스 검색 실패: {e}")
            return []
    
    def _combine_search_results(
        self,
        vector_results: List[SearchResult],
        graph_results: List[Dict]
    ) -> List[Dict]:
        """검색 결과 통합"""
        combined = {}
        
        # 벡터 검색 결과 처리
        for result in vector_results:
            file_path = result.chunk.file_path
            if file_path not in combined:
                combined[file_path] = {
                    'similarity_score': result.similarity_score,
                    'relevance_score': result.relevance_score,
                    'graph_connections': result.graph_connections or [],
                    'chunks': []
                }
            combined[file_path]['chunks'].append(result.chunk)
        
        # 그래프 검색 결과 처리
        for result in graph_results:
            file_path = result['path']
            if file_path not in combined:
                combined[file_path] = {
                    'similarity_score': 0.0,
                    'relevance_score': result.get('weight', 0.0),
                    'graph_connections': result.get('types', []),
                    'chunks': []
                }
            else:
                combined[file_path]['graph_connections'].extend(result.get('types', []))
                combined[file_path]['relevance_score'] = max(
                    combined[file_path]['relevance_score'],
                    result.get('weight', 0.0)
                )
        
        return list(combined.values())
    
    def _rank_results(self, results: List[Dict]) -> List[Dict]:
        """결과 정렬"""
        def calculate_score(result: Dict) -> float:
            similarity_weight = 0.6
            relevance_weight = 0.4
            
            similarity_score = result['similarity_score']
            relevance_score = result['relevance_score']
            
            # 그래프 연결성 점수 계산
            graph_score = len(result['graph_connections']) / 10.0
            
            return (
                similarity_weight * similarity_score +
                relevance_weight * relevance_score +
                graph_score
            )
        
        return sorted(results, key=calculate_score, reverse=True)
    
    async def get_context_for_query(
        self,
        query: str,
        max_context_length: int = 4000
    ) -> str:
        """쿼리에 대한 컨텍스트 생성"""
        try:
            # 코드베이스 검색
            results = await self.search_codebase(query)
            
            if not results:
                return ""
            
            # 컨텍스트 생성
            context = []
            current_length = 0
            
            for result in results:
                for chunk in result['chunks']:
                    chunk_text = f"File: {chunk.file_path}\n{chunk.content}\n"
                    if current_length + len(chunk_text) > max_context_length:
                        break
                    context.append(chunk_text)
                    current_length += len(chunk_text)
            
            return "\n".join(context)
            
        except Exception as e:
            logger.error(f"컨텍스트 생성 실패: {e}")
            return ""
    
    async def answer_with_context(self, query: str, context: str = None) -> str:
        """컨텍스트를 활용한 답변 생성"""
        try:
            if not context:
                context = await self.get_context_for_query(query)
            
            if not context:
                return "관련 코드를 찾을 수 없습니다."
            
            # LLM을 사용한 답변 생성
            prompt = f"""다음은 코드베이스에서 검색된 관련 코드입니다:

{context}

질문: {query}

위 코드를 바탕으로 질문에 답변해주세요."""

            response = await self.llm_manager.generate(prompt)
            return response
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return "답변을 생성하는 중 오류가 발생했습니다."

class CodeIndexer:
    """코드 인덱서"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self._indexing_stats = {
            'total_files': 0,
            'total_chunks': 0,
            'last_full_index': None,
            'indexing_time': 0
        }
        
    def get_indexing_stats(self) -> Dict[str, Any]:
        """인덱싱 통계 반환"""
        return self._indexing_stats
        
    async def index_directory(self, directory: str, max_files: int = 1000) -> Dict[str, Any]:
        """디렉토리 인덱싱"""
        try:
            start_time = time.time()
            total_files = 0
            success_count = 0
            
            # 디렉토리 내 모든 파일 수집
            for root, _, files in os.walk(directory):
                for file in files:
                    if total_files >= max_files:
                        break
                        
                    file_path = os.path.join(root, file)
                    if not self._is_supported_file(file_path):
                        continue
                        
                    total_files += 1
                    try:
                        # 파일 분석
                        analysis = await self.rag_system.code_analyzer.analyze_file(file_path)
                        if not analysis:
                            continue
                            
                        # 청크 생성
                        chunks = self._create_chunks(file_path, analysis)
                        if not chunks:
                            continue
                            
                        # 벡터 DB에 추가
                        if await self.rag_system.vector_db.add_chunks(chunks):
                            success_count += 1
                            
                    except Exception as e:
                        logger.error(f"파일 인덱싱 실패: {file_path}: {e}")
                        
            # 통계 업데이트
            self._indexing_stats.update({
                'total_files': total_files,
                'total_chunks': len(self.rag_system.vector_db.chunk_map),
                'last_full_index': datetime.now().isoformat(),
                'indexing_time': time.time() - start_time
            })
            
            return {
                'status': 'success',
                'total_files': total_files,
                'success_count': success_count,
                'indexing_time': self._indexing_stats['indexing_time']
            }
            
        except Exception as e:
            logger.error(f"디렉토리 인덱싱 실패: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def _is_supported_file(self, file_path: str) -> bool:
        """지원되는 파일인지 확인"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in {'.py', '.java', '.xml', '.js', '.ts', '.cpp', '.c', '.h', '.hpp'}
        
    def _create_chunks(self, file_path: str, analysis: Dict[str, Any]) -> List[CodeChunk]:
        """분석 결과로부터 청크 생성"""
        chunks = []
        
        # 함수 청크
        for func in analysis.get('functions', []):
            chunk = CodeChunk(
                id=f"{file_path}:{func['start_line']}:{func['end_line']}",
                file_path=file_path,
                content=func['content'],
                start_line=func['start_line'],
                end_line=func['end_line'],
                language=analysis.get('language', ''),
                chunk_type='function',
                metadata=func
            )
            chunks.append(chunk)
            
        # 클래스 청크
        for cls in analysis.get('classes', []):
            chunk = CodeChunk(
                id=f"{file_path}:{cls['start_line']}:{cls['end_line']}",
                file_path=file_path,
                content=cls['content'],
                start_line=cls['start_line'],
                end_line=cls['end_line'],
                language=analysis.get('language', ''),
                chunk_type='class',
                metadata=cls
            )
            chunks.append(chunk)
            
        # 임포트 청크
        for imp in analysis.get('imports', []):
            chunk = CodeChunk(
                id=f"{file_path}:import:{imp['name']}",
                file_path=file_path,
                content=imp['content'],
                start_line=imp['start_line'],
                end_line=imp['end_line'],
                language=analysis.get('language', ''),
                chunk_type='import',
                metadata=imp
            )
            chunks.append(chunk)
            
        return chunks

def get_rag_system() -> RAGSystem:
    """RAG 시스템 인스턴스 반환"""
    return RAGSystem()