#!/usr/bin/env python3
"""
Open CodeAI 시스템 검증 스크립트
전체 시스템 상태를 확인하고 문제점을 진단
"""
import os
import sys
import asyncio
import argparse
import json
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_basic_imports():
    """기본 import 확인"""
    try:
        from src.utils.validation import run_comprehensive_validation, generate_validation_report
        from src.config import settings
        from src.utils.logger import get_logger
        return True
    except ImportError as e:
        print(f"❌ 기본 모듈 import 실패: {e}")
        print("Python 가상환경이 활성화되어 있는지 확인하세요.")
        return False

async def run_detailed_checks():
    """상세 시스템 검사 실행"""
    try:
        from src.utils.validation import run_comprehensive_validation, generate_validation_report
        from src.utils.logger import get_logger
        
        logger = get_logger(__name__)
        
        print("🔍 종합 시스템 검증 실행 중...")
        print("이 과정은 몇 분 소요될 수 있습니다...")
        print()
        
        # 종합 검증 실행
        validation_results = run_comprehensive_validation()
        
        # 리포트 생성
        report = generate_validation_report(validation_results)
        
        # 리포트 출력
        print(report)
        
        # 파일로 저장
        report_path = "logs/system_validation_report.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 상세 리포트가 저장되었습니다: {report_path}")
        
        # JSON 형태로도 저장
        json_path = "logs/system_validation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 JSON 결과가 저장되었습니다: {json_path}")
        
        return validation_results
        
    except Exception as e:
        print(f"❌ 종합 검증 실패: {e}")
        return None

def check_quick_status():
    """빠른 상태 확인"""
    print("⚡ 빠른 시스템 상태 확인")
    print("-" * 40)
    
    checks = [
        ("Python 버전", check_python_version),
        ("필수 패키지", check_essential_packages),
        ("디렉토리 구조", check_directory_structure),
        ("설정 파일", check_config_files),
        ("Docker 서비스", check_docker_services),
        ("API 서버", check_api_server)
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            result = check_func()
            status = "✅" if result else "❌"
            print(f"{status} {name}")
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {name} (오류: {e})")
    
    print("-" * 40)
    print(f"📊 결과: {passed}/{total} 항목 통과")
    
    if passed == total:
        print("🎉 모든 기본 검사를 통과했습니다!")
        return True
    elif passed >= total * 0.7:
        print("⚠️ 대부분의 검사를 통과했지만 일부 문제가 있습니다.")
        return False
    else:
        print("❌ 여러 문제가 발견되었습니다. 상세 검사를 실행하세요.")
        return False

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    required = (3, 10)
    return version >= required

def check_essential_packages():
    """필수 패키지 확인"""
    packages = ['fastapi', 'uvicorn', 'pydantic', 'yaml', 'loguru']
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            return False
    return True

def check_directory_structure():
    """디렉토리 구조 확인"""
    required_dirs = [
        'src', 'src/api', 'src/core', 'src/utils',
        'configs', 'data', 'logs', 'scripts'
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            return False
    return True

def check_config_files():
    """설정 파일 확인"""
    config_files = [
        'configs/config.yaml',
        '.env.example'
    ]
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            return False
    return True

def check_docker_services():
    """Docker 서비스 확인"""
    try:
        import subprocess
        result = subprocess.run(['docker', 'ps'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def check_api_server():
    """API 서버 확인"""
    try:
        import requests
        response = requests.get('http://localhost:8000/v1/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def show_troubleshooting_guide():
    """문제 해결 가이드 표시"""
    print("\n🔧 문제 해결 가이드")
    print("=" * 50)
    
    guides = [
        {
            "문제": "Python 가상환경 활성화",
            "해결": [
                "source venv/bin/activate  # Linux/Mac",
                "venv\\Scripts\\activate   # Windows"
            ]
        },
        {
            "문제": "필수 패키지 설치",
            "해결": [
                "pip install -r requirements.txt",
                "pip install -r requirements-dev.txt  # 개발 환경"
            ]
        },
        {
            "문제": "Docker 서비스 시작",
            "해결": [
                "sudo systemctl start docker  # Linux",
                "# Docker Desktop 시작  # Windows/Mac",
                "docker-compose up -d  # 컨테이너 시작"
            ]
        },
        {
            "문제": "모델 다운로드",
            "해결": [
                "python scripts/download_models.py",
                "python scripts/download_models.py --required-only"
            ]
        },
        {
            "문제": "데이터베이스 초기화",
            "해결": [
                "python scripts/init_databases.py",
                "docker-compose restart neo4j"
            ]
        },
        {
            "문제": "Continue.dev 설정",
            "해결": [
                "cp examples/continue_config.json ~/.continue/config.json",
                "VS Code에서 Continue 확장 재시작"
            ]
        }
    ]
    
    for i, guide in enumerate(guides, 1):
        print(f"\n{i}. {guide['문제']}")
        for solution in guide['해결']:
            print(f"   {solution}")
    
    print("\n📚 추가 도움말:")
    print("   - 상세 로그: tail -f logs/opencodeai.log")
    print("   - 설치 가이드: README.md")
    print("   - 이슈 리포트: https://github.com/ChangooLee/open-codeai/issues")

def generate_system_info():
    """시스템 정보 수집"""
    info = {
        "timestamp": time.time(),
        "python_version": sys.version,
        "platform": sys.platform,
        "working_directory": os.getcwd(),
        "environment_variables": {},
        "installed_packages": [],
        "file_structure": {}
    }
    
    # 환경 변수 수집 (민감한 정보 제외)
    sensitive_keys = ['password', 'key', 'secret', 'token']
    for key, value in os.environ.items():
        if not any(sensitive in key.lower() for sensitive in sensitive_keys):
            info["environment_variables"][key] = value
    
    # 설치된 패키지 정보
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            info["installed_packages"] = json.loads(result.stdout)
    except:
        pass
    
    # 파일 구조 확인
    important_paths = [
        'src', 'configs', 'data', 'logs', 'scripts',
        'requirements.txt', '.env', 'docker-compose.yml'
    ]
    
    for path in important_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                info["file_structure"][path] = "file"
            else:
                info["file_structure"][path] = "directory"
        else:
            info["file_structure"][path] = "missing"
    
    return info

def main():
    parser = argparse.ArgumentParser(description="Open CodeAI 시스템 검증")
    parser.add_argument("--quick", action="store_true", help="빠른 상태 확인만 실행")
    parser.add_argument("--detailed", action="store_true", help="상세 검증 실행")
    parser.add_argument("--troubleshoot", action="store_true", help="문제 해결 가이드 표시")
    parser.add_argument("--system-info", action="store_true", help="시스템 정보 수집")
    parser.add_argument("--export", type=str, help="결과를 파일로 내보내기")
    
    args = parser.parse_args()
    
    print("🔍 Open CodeAI 시스템 검증")
    print("=" * 50)
    
    # 기본 import 확인
    if not check_basic_imports():
        print("\n❌ 기본 모듈을 import할 수 없습니다.")
        print("다음을 확인하세요:")
        print("1. Python 가상환경이 활성화되어 있는지")
        print("2. 현재 디렉토리가 프로젝트 루트인지") 
        print("3. 필수 패키지가 설치되어 있는지")
        sys.exit(1)
    
    # 시스템 정보 수집
    if args.system_info:
        print("\n📊 시스템 정보 수집 중...")
        import time
        system_info = generate_system_info()
        
        if args.export:
            with open(args.export, 'w', encoding='utf-8') as f:
                json.dump(system_info, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ 시스템 정보가 저장되었습니다: {args.export}")
        else:
            print(json.dumps(system_info, indent=2, ensure_ascii=False, default=str))
        return
    
    # 문제 해결 가이드
    if args.troubleshoot:
        show_troubleshooting_guide()
        return
    
    # 빠른 검사
    if args.quick or (not args.detailed):
        quick_result = check_quick_status()
        
        if not quick_result and not args.quick:
            print("\n💡 더 자세한 정보가 필요하면 다음 명령을 실행하세요:")
            print("python scripts/system_check.py --detailed")
        
        return
    
    # 상세 검사
    if args.detailed:
        try:
            validation_results = asyncio.run(run_detailed_checks())
            
            if validation_results:
                overall_status = validation_results.get('overall_status', 'unknown')
                
                if overall_status == 'healthy':
                    print("\n🎉 시스템이 정상적으로 구성되었습니다!")
                    print("Open CodeAI를 시작할 수 있습니다: ./start.sh")
                elif overall_status in ['caution', 'warning']:
                    print("\n⚠️ 시스템에 일부 문제가 있지만 실행 가능합니다.")
                    print("권장사항을 확인하여 성능을 개선할 수 있습니다.")
                else:
                    print("\n❌ 시스템에 중요한 문제가 있습니다.")
                    print("권장사항을 따라 문제를 해결하세요.")
                    
                    # 문제 해결 가이드 표시
                    show_troubleshooting_guide()
                
                # 결과 내보내기
                if args.export:
                    with open(args.export, 'w', encoding='utf-8') as f:
                        json.dump(validation_results, f, indent=2, ensure_ascii=False, default=str)
                    print(f"\n📄 상세 결과가 저장되었습니다: {args.export}")
            
        except KeyboardInterrupt:
            print("\n👋 검증이 중단되었습니다.")
        except Exception as e:
            print(f"\n❌ 상세 검증 실행 중 오류: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()