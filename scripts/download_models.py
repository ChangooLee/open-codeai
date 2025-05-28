#!/usr/bin/env python3
"""
Open CodeAI 모델 다운로드 스크립트
HuggingFace Hub에서 필요한 AI 모델들을 다운로드
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
import subprocess

def install_huggingface_hub():
    """HuggingFace Hub 패키지 설치"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("📦 HuggingFace Hub 패키지 설치 중...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            import huggingface_hub
            return True
        except Exception as e:
            print(f"❌ HuggingFace Hub 설치 실패: {e}")
            return False

def format_size(bytes_size: int) -> str:
    """바이트를 읽기 좋은 형식으로 변환"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"

def get_model_info(repo_id: str) -> Dict[str, Any]:
    """모델 정보 조회"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # 모델 정보 조회
        model_info = api.model_info(repo_id)
        
        # 파일 크기 계산
        total_size = 0
        if hasattr(model_info, 'siblings') and model_info.siblings:
            for file_info in model_info.siblings:
                if hasattr(file_info, 'size') and file_info.size:
                    total_size += file_info.size
        
        return {
            'repo_id': repo_id,
            'total_size': total_size,
            'formatted_size': format_size(total_size),
            'files_count': len(model_info.siblings) if hasattr(model_info, 'siblings') else 0,
            'model_info': model_info
        }
    except Exception as e:
        return {
            'repo_id': repo_id,
            'error': str(e),
            'total_size': 0,
            'formatted_size': 'Unknown',
            'files_count': 0
        }

def download_model(repo_id: str, local_dir: str, model_name: str, skip_existing: bool = True) -> bool:
    """단일 모델 다운로드"""
    try:
        from huggingface_hub import snapshot_download
        
        local_path = Path(local_dir)
        
        # 이미 존재하는 경우 건너뛰기 옵션
        if skip_existing and local_path.exists():
            required_files = ['config.json']
            model_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.safetensors']
            
            # 필수 파일 확인
            has_config = (local_path / 'config.json').exists()
            has_model = any((local_path / f).exists() for f in model_files)
            
            if has_config and has_model:
                print(f"✅ {model_name} 이미 다운로드됨, 건너뜀")
                return True
        
        print(f"📥 {model_name} 다운로드 시작...")
        print(f"   저장소: {repo_id}")
        print(f"   저장 경로: {local_dir}")
        
        # 모델 정보 조회
        model_info = get_model_info(repo_id)
        if 'error' not in model_info:
            print(f"   예상 크기: {model_info['formatted_size']}")
        
        # 디렉토리 생성
        local_path.mkdir(parents=True, exist_ok=True)
        
        # 다운로드 진행
        start_time = time.time()
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            cache_dir="./data/cache/huggingface",
            resume_download=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ {model_name} 다운로드 완료 ({elapsed_time:.1f}초)")
        
        # 다운로드 확인
        if (local_path / 'config.json').exists():
            return True
        else:
            print(f"❌ {model_name} 다운로드 후 필수 파일이 없습니다")
            return False
        
    except Exception as e:
        print(f"❌ {model_name} 다운로드 실패: {e}")
        return False

def load_models_config(config_path: str) -> Dict[str, Any]:
    """모델 설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        return {}

def get_default_models() -> List[Dict[str, Any]]:
    """기본 모델 목록 반환"""
    return [
        {
            "repo_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "local_dir": "./data/models/qwen2.5-coder-32b",
            "name": "Qwen2.5-Coder-32B (메인 LLM)",
            "description": "코드 생성 및 분석용 메인 언어 모델",
            "size_estimate": "64GB",
            "priority": 1,
            "required": True
        },
        {
            "repo_id": "BAAI/bge-large-en-v1.5",
            "local_dir": "./data/models/bge-large-en-v1.5",
            "name": "BGE Large (임베딩 모델)",
            "description": "텍스트 임베딩 생성용 모델",
            "size_estimate": "2.3GB",
            "priority": 2,
            "required": True
        },
        {
            "repo_id": "Salesforce/codet5-small",
            "local_dir": "./data/models/codet5-small",
            "name": "CodeT5 Small (그래프 분석)",
            "description": "코드 구조 분석용 모델",
            "size_estimate": "242MB",
            "priority": 3,
            "required": False
        },
        {
            "repo_id": "microsoft/codebert-base",
            "local_dir": "./data/models/codebert-base",
            "name": "CodeBERT (코드 이해)",
            "description": "코드 이해 및 검색용 모델",
            "size_estimate": "500MB",
            "priority": 4,
            "required": False
        }
    ]

def check_disk_space(required_gb: float) -> bool:
    """디스크 공간 확인"""
    try:
        import shutil
        free_space = shutil.disk_usage('.').free
        free_gb = free_space / (1024**3)
        
        print(f"💿 사용 가능한 디스크 공간: {free_gb:.1f}GB")
        print(f"📦 필요한 공간: {required_gb:.1f}GB")
        
        if free_gb < required_gb:
            print(f"❌ 디스크 공간 부족 ({free_gb:.1f}GB < {required_gb:.1f}GB)")
            return False
        
        return True
    except Exception as e:
        print(f"⚠️ 디스크 공간 확인 실패: {e}")
        return True  # 확인 실패시 계속 진행

def estimate_download_time(total_size_gb: float, speed_mbps: float = 50) -> str:
    """다운로드 시간 추정"""
    total_size_mb = total_size_gb * 1024
    time_minutes = total_size_mb / speed_mbps / 60 * 8  # 8 bits per byte
    
    if time_minutes < 60:
        return f"{time_minutes:.0f}분"
    else:
        hours = time_minutes / 60
        return f"{hours:.1f}시간"

def main():
    parser = argparse.ArgumentParser(description="Open CodeAI 모델 다운로드")
    parser.add_argument("--config", type=str, help="모델 설정 파일 경로")
    parser.add_argument("--model", type=str, action='append', help="특정 모델만 다운로드 (여러 번 사용 가능)")
    parser.add_argument("--skip-existing", action='store_true', default=True, help="이미 다운로드된 모델 건너뛰기")
    parser.add_argument("--force", action='store_true', help="기존 모델도 재다운로드")
    parser.add_argument("--list-only", action='store_true', help="다운로드할 모델 목록만 표시")
    parser.add_argument("--required-only", action='store_true', help="필수 모델만 다운로드")
    parser.add_argument("--cache-dir", type=str, default="./data/cache/huggingface", help="캐시 디렉토리")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🤖 Open CodeAI 모델 다운로드 시작")
    print("=" * 70)
    
    # HuggingFace Hub 설치 확인
    if not install_huggingface_hub():
        sys.exit(1)
    
    # 모델 목록 로드
    if args.config and os.path.exists(args.config):
        print(f"📋 설정 파일에서 모델 목록 로드: {args.config}")
        config_data = load_models_config(args.config)
        models = config_data.get('models', [])
        if not models:
            print("⚠️ 설정 파일에 모델 정보가 없습니다. 기본 모델 사용")
            models = get_default_models()
    else:
        print("📋 기본 모델 목록 사용")
        models = get_default_models()
    
    # 특정 모델 필터링
    if args.model:
        models = [m for m in models if any(name in m.get('name', '') or name in m.get('repo_id', '') for name in args.model)]
        if not models:
            print(f"❌ 지정된 모델을 찾을 수 없습니다: {args.model}")
            sys.exit(1)
    
    # 필수 모델만 필터링
    if args.required_only:
        models = [m for m in models if m.get('required', True)]
    
    # Force 옵션 처리
    if args.force:
        args.skip_existing = False
    
    # 우선순위로 정렬
    models.sort(key=lambda x: x.get('priority', 999))
    
    print(f"\n📦 다운로드할 모델: {len(models)}개")
    print("-" * 50)
    
    total_size_gb = 0
    for i, model in enumerate(models, 1):
        name = model.get('name', model.get('repo_id', 'Unknown'))
        size_est = model.get('size_estimate', 'Unknown')
        required = "✅ 필수" if model.get('required', True) else "🔵 선택"
        description = model.get('description', '')
        
        print(f"{i}. {name}")
        print(f"   크기: {size_est} | {required}")
        if description:
            print(f"   설명: {description}")
        print()
        
        # 크기 추정치 누적
        if size_est != 'Unknown':
            size_num = float(size_est.replace('GB', '').replace('MB', ''))
            if 'GB' in size_est:
                total_size_gb += size_num
            elif 'MB' in size_est:
                total_size_gb += size_num / 1024
    
    print(f"💾 총 예상 크기: {total_size_gb:.1f}GB")
    print(f"⏱️  예상 다운로드 시간: {estimate_download_time(total_size_gb)}")
    
    # 리스트만 표시하는 경우
    if args.list_only:
        print("\n📋 다운로드할 모델 목록이 표시되었습니다.")
        sys.exit(0)
    
    # 디스크 공간 확인
    if not check_disk_space(total_size_gb * 1.2):  # 20% 여유 공간
        response = input("\n디스크 공간이 부족할 수 있습니다. 계속하시겠습니까? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("다운로드를 취소합니다.")
            sys.exit(0)
    
    # 확인 메시지
    print("\n⚠️ 참고사항:")
    print("- 인터넷 연결이 필요합니다")
    print("- 다운로드 중 중단된 경우 재시작하면 이어서 받습니다")
    print("- 모델은 ./data/models/ 디렉토리에 저장됩니다")
    print()
    
    response = input("다운로드를 시작하시겠습니까? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("다운로드를 취소합니다.")
        sys.exit(0)
    
    print("\n🚀 모델 다운로드 시작!")
    print("=" * 50)
    
    # 캐시 디렉토리 생성
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 다운로드 진행
    success_count = 0
    start_time = time.time()
    
    for i, model in enumerate(models, 1):
        repo_id = model.get('repo_id')
        local_dir = model.get('local_dir')
        name = model.get('name', repo_id)
        
        if not repo_id or not local_dir:
            print(f"❌ {name}: 설정 정보 부족")
            continue
        
        print(f"\n[{i}/{len(models)}] {name}")
        print("-" * 30)
        
        if download_model(repo_id, local_dir, name, args.skip_existing):
            success_count += 1
        else:
            # 필수 모델 다운로드 실패 시 중단 여부 확인
            if model.get('required', True):
                response = input(f"필수 모델 다운로드 실패. 계속하시겠습니까? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    break
    
    # 결과 요약
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("📊 다운로드 완료!")
    print("=" * 70)
    print(f"✅ 성공: {success_count}/{len(models)} 모델")
    print(f"⏱️  소요 시간: {elapsed_time/60:.1f}분")
    print()
    
    if success_count == len(models):
        print("🎉 모든 모델이 성공적으로 다운로드되었습니다!")
        print("이제 Open CodeAI를 시작할 수 있습니다: ./start.sh")
    elif success_count > 0:
        print("⚠️ 일부 모델만 다운로드되었습니다.")
        print("필수 모델이 있다면 Open CodeAI를 시작할 수 있습니다.")
        print("누락된 모델은 나중에 다시 다운로드할 수 있습니다.")
    else:
        print("❌ 모델 다운로드에 실패했습니다.")
        print("인터넷 연결과 디스크 공간을 확인하세요.")
        sys.exit(1)
    
    print("\n💡 다음 단계:")
    print("1. 서버 시작: ./start.sh")
    print("2. 프로젝트 인덱싱: ./index.sh /path/to/project")
    print("3. Continue.dev 확장 설치 및 설정")

if __name__ == "__main__":
    main()