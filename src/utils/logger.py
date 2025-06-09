"""
로깅 시스템 설정
"""
import sys
from pathlib import Path
from typing import Optional, Callable, Any
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def setup_logger(
    name: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True
) -> None:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        log_level: 로그 레벨
        log_dir: 로그 파일 저장 디렉토리
        enable_file_logging: 파일 로깅 활성화 여부
    """
    
    # 기존 핸들러 제거
    logger.remove()
    
    # 콘솔 로깅 설정
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="DEBUG",
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    if enable_file_logging:
        # 로그 디렉토리 생성
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 일반 로그 파일
        logger.add(
            log_path / "opencodeai.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="50 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # 에러 로그 파일
        logger.add(
            log_path / "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="10 MB", 
            retention="90 days",
            compression="gz",
            backtrace=True,
            diagnose=True
        )
        
        # 성능 로그 파일
        logger.add(
            log_path / "performance.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "PERF" in record["message"],
            rotation="10 MB",
            retention="7 days"
        )


def get_logger(name: str) -> Any:
    """
    이름이 지정된 로거 반환
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거 인스턴스
    """
    return logger.bind(name=name)


# 성능 측정용 데코레이터
def log_performance(func_name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """함수 실행 시간 로깅 데코레이터"""
    def decorator(func: Callable) -> Callable:
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            name = func_name or func.__name__
            logger.info(f"PERF | {name} executed in {execution_time:.4f}s")
            
            return result
        return wrapper
    return decorator
# 기본 로거 설정 (모듈 로드 시 자동 실행)
try:
    from src.config import settings
    setup_logger(
        log_level=getattr(settings, 'LOG_LEVEL', 'INFO'),
        enable_file_logging=True
    )
except ImportError:
    # 설정 파일이 없을 경우 기본 설정 사용
    setup_logger(log_level="INFO")


