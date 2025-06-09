"""
Open CodeAI - 시스템 검증 및 진단 모듈
완전한 시스템 상태 확인 및 문제 진단 기능
"""
import os
import sys
import json
import time
import asyncio
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import platform
import psutil

def validate_model_path(path: str) -> bool:
    """모델 경로 검증"""
    try:
        model_path = Path(path)
        if not model_path.exists():
            return False
        
        # 필수 모델 파일들 확인
        required_files = ['config.json']
        optional_files = [
            'pytorch_model.bin', 'model.safetensors', 
            'pytorch_model.safetensors', 'tokenizer.json',
            'tokenizer_config.json', 'vocab.txt'
        ]
        
        # config.json은 필수
        if not (model_path / 'config.json').exists():
            return False
        
        # 최소 하나의 모델 파일은 있어야 함
        has_model_file = any(
            (model_path / file).exists() for file in optional_files
        )
        
        return has_model_file
        
    except Exception:
        return False

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """설정 파일 검증"""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # 필수 섹션 확인
        required_sections = ['project', 'server']
        for section in required_sections:
            if section not in config:
                validation_result['errors'].append(f"필수 섹션 누락: {section}")
                validation_result['valid'] = False
        
        # 프로젝트 설정 검증
        if 'project' in config:
            project = config['project']
            if 'max_files' in project:
                max_files = project['max_files']
                if max_files > 50000:
                    validation_result['warnings'].append(
                        f"max_files가 매우 큼 ({max_files}). 성능에 영향을 줄 수 있습니다."
                    )
                elif max_files < 100:
                    validation_result['warnings'].append(
                        f"max_files가 작음 ({max_files}). 대형 프로젝트에 부족할 수 있습니다."
                    )
        
        # 서버 설정 검증
        if 'server' in config:
            server = config['server']
            if 'port' in server:
                port = server['port']
                if port < 1024:
                    validation_result['warnings'].append(
                        f"포트 {port}는 관리자 권한이 필요할 수 있습니다."
                    )
                elif port > 65535:
                    validation_result['errors'].append(f"잘못된 포트 번호: {port}")
                    validation_result['valid'] = False
        
        # LLM 설정 검증
        if 'llm' in config:
            llm = config['llm']
            if 'main_model' in llm:
                main_model = llm['main_model']
                if 'path' in main_model:
                    model_path = main_model['path']
                    if not validate_model_path(model_path):
                        validation_result['warnings'].append(
                            f"메인 모델 경로 확인 필요: {model_path}"
                        )
                        validation_result['recommendations'].append(
                            "python scripts/download_models.py 실행을 권장합니다."
                        )
        
        # 성능 권장사항
        if 'performance' in config:
            perf = config['performance']
            if 'gpu' in perf and perf['gpu'].get('enable', False):
                # GPU 실제 사용 가능 여부 확인
                try:
                    import torch
                    if not torch.cuda.is_available():
                        validation_result['warnings'].append(
                            "GPU가 활성화되어 있지만 CUDA를 사용할 수 없습니다."
                        )
                except ImportError:
                    validation_result['errors'].append(
                        "GPU 설정이 활성화되어 있지만 PyTorch가 설치되지 않았습니다."
                    )
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"설정 검증 중 오류: {str(e)}")
    
    return validation_result

def check_system_requirements() -> Dict[str, Any]:
    """시스템 요구사항 확인"""
    requirements = {
        'python_version': {
            'required': '3.10+',
            'current': platform.python_version(),
            'status': 'unknown'
        },
        'memory': {
            'required_gb': 16,
            'recommended_gb': 32,
            'current_gb': 0,
            'status': 'unknown'
        },
        'disk_space': {
            'required_gb': 50,
            'recommended_gb': 200,
            'current_gb': 0,
            'status': 'unknown'
        },
        'cpu_cores': {
            'required': 4,
            'recommended': 8,
            'current': 0,
            'status': 'unknown'
        },
        'gpu': {
            'required': False,
            'available': False,
            'devices': [],
            'status': 'unknown'
        }
    }
    
    try:
        # Python 버전 확인
        version_info = sys.version_info
        if version_info >= (3, 10):
            requirements['python_version']['status'] = 'ok'
        elif version_info >= (3, 8):
            requirements['python_version']['status'] = 'warning'
        else:
            requirements['python_version']['status'] = 'error'
        
        # 메모리 확인
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        requirements['memory']['current_gb'] = round(memory_gb, 1)
        
        if memory_gb >= 32:
            requirements['memory']['status'] = 'excellent'
        elif memory_gb >= 16:
            requirements['memory']['status'] = 'ok'
        elif memory_gb >= 8:
            requirements['memory']['status'] = 'warning'
        else:
            requirements['memory']['status'] = 'error'
        
        # 디스크 공간 확인
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        requirements['disk_space']['current_gb'] = round(disk_gb, 1)
        
        if disk_gb >= 200:
            requirements['disk_space']['status'] = 'excellent'
        elif disk_gb >= 50:
            requirements['disk_space']['status'] = 'ok'
        elif disk_gb >= 20:
            requirements['disk_space']['status'] = 'warning'
        else:
            requirements['disk_space']['status'] = 'error'
        
        # CPU 확인
        cpu_count = psutil.cpu_count(logical=False)
        requirements['cpu_cores']['current'] = cpu_count
        
        if cpu_count >= 16:
            requirements['cpu_cores']['status'] = 'excellent'
        elif cpu_count >= 8:
            requirements['cpu_cores']['status'] = 'ok'
        elif cpu_count >= 4:
            requirements['cpu_cores']['status'] = 'warning'
        else:
            requirements['cpu_cores']['status'] = 'error'
        
        # GPU 확인
        try:
            import torch
            if torch.cuda.is_available():
                requirements['gpu']['available'] = True
                requirements['gpu']['status'] = 'ok'
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    requirements['gpu']['devices'].append({
                        'id': i,
                        'name': props.name,
                        'memory_gb': round(props.total_memory / (1024**3), 1)
                    })
            else:
                requirements['gpu']['status'] = 'warning'
        except ImportError:
            requirements['gpu']['status'] = 'warning'
    
    except Exception as e:
        print(f"시스템 요구사항 확인 오류: {e}")
    
    return requirements

def check_dependencies() -> Dict[str, Any]:
    """의존성 패키지 확인"""
    dependencies = {
        'critical': [
            'fastapi', 'uvicorn', 'pydantic', 'pydantic-settings'
        ],
        'important': [
            'torch', 'transformers', 'sentence-transformers'
        ],
        'optional': [
            'faiss-cpu', 'faiss-gpu', 'neo4j', 'networkx'
        ],
        'development': [
            'pytest', 'black', 'isort', 'mypy'
        ]
    }
    
    results = {
        'critical': {'installed': [], 'missing': []},
        'important': {'installed': [], 'missing': []},
        'optional': {'installed': [], 'missing': []},
        'development': {'installed': [], 'missing': []},
        'status': 'unknown'
    }
    
    for category, packages in dependencies.items():
        for package in packages:
            try:
                # 패키지명 정규화
                import_name = package.replace('-', '_')
                if import_name == 'pydantic_settings':
                    import_name = 'pydantic_settings'
                elif import_name == 'sentence_transformers':
                    import_name = 'sentence_transformers'
                
                __import__(import_name)
                results[category]['installed'].append(package)
            except ImportError:
                results[category]['missing'].append(package)
    
    # 전체 상태 결정
    if results['critical']['missing']:
        results['status'] = 'error'
    elif results['important']['missing']:
        results['status'] = 'warning'
    else:
        results['status'] = 'ok'
    
    return results

def check_services() -> Dict[str, Any]:
    """외부 서비스 상태 확인"""
    services = {
        'docker': {'status': 'unknown', 'version': None},
        'neo4j': {'status': 'unknown', 'accessible': False},
        'api_server': {'status': 'unknown', 'accessible': False}
    }
    
    # Docker 확인
    try:
        result = subprocess.run(
            ['docker', '--version'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            services['docker']['status'] = 'ok'
            services['docker']['version'] = result.stdout.strip()
        else:
            services['docker']['status'] = 'error'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        services['docker']['status'] = 'not_installed'
    
    # Neo4j 확인
    try:
        import requests
        response = requests.get(
            'http://localhost:7474',
            timeout=5
        )
        if response.status_code == 200:
            services['neo4j']['status'] = 'ok'
            services['neo4j']['accessible'] = True
        else:
            services['neo4j']['status'] = 'error'
    except Exception:
        services['neo4j']['status'] = 'not_running'
    
    # API 서버 확인
    try:
        import requests
        response = requests.get(
            'http://localhost:8800/v1/health',
            timeout=5
        )
        if response.status_code == 200:
            services['api_server']['status'] = 'ok'
            services['api_server']['accessible'] = True
        else:
            services['api_server']['status'] = 'error'
    except Exception:
        services['api_server']['status'] = 'not_running'
    
    return services

def check_file_structure() -> Dict[str, Any]:
    """파일 구조 확인"""
    required_structure = {
        'directories': [
            'src', 'src/api', 'src/core', 'src/utils',
            'data', 'logs', 'scripts'
        ],
        'files': [
            'src/main.py', 'src/config.py',
            'requirements.txt', '.env.example',
            'docker-compose.yml'
        ],
        'optional_directories': [
            'data/models', 'data/vector_index', 'data/graph_db',
            'data/metadata', 'tests', 'static'
        ]
    }
    
    results = {
        'directories': {'present': [], 'missing': []},
        'files': {'present': [], 'missing': []},
        'optional_directories': {'present': [], 'missing': []},
        'status': 'unknown'
    }
    
    # 필수 디렉토리 확인
    for directory in required_structure['directories']:
        if os.path.isdir(directory):
            results['directories']['present'].append(directory)
        else:
            results['directories']['missing'].append(directory)
    
    # 필수 파일 확인
    for file_path in required_structure['files']:
        if os.path.isfile(file_path):
            results['files']['present'].append(file_path)
        else:
            results['files']['missing'].append(file_path)
    
    # 선택적 디렉토리 확인
    for directory in required_structure['optional_directories']:
        if os.path.isdir(directory):
            results['optional_directories']['present'].append(directory)
        else:
            results['optional_directories']['missing'].append(directory)
    
    # 전체 상태 결정
    if results['directories']['missing'] or results['files']['missing']:
        if len(results['directories']['missing']) > 2 or len(results['files']['missing']) > 2:
            results['status'] = 'error'
        else:
            results['status'] = 'warning'
    else:
        results['status'] = 'ok'
    
    return results

def run_comprehensive_validation() -> Dict[str, Any]:
    """종합 시스템 검증 실행"""
    print("🔍 Open CodeAI 종합 시스템 검증 시작...")
    start_time = time.time()
    
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'unknown',
        'summary': {},
        'details': {},
        'recommendations': [],
        'execution_time': 0
    }
    
    try:
        # 1. 시스템 요구사항 확인
        print("   📋 시스템 요구사항 확인 중...")
        validation_results['details']['system_requirements'] = check_system_requirements()
        
        # 2. 의존성 패키지 확인
        print("   📦 의존성 패키지 확인 중...")
        validation_results['details']['dependencies'] = check_dependencies()
        
        # 3. 파일 구조 확인
        print("   📁 파일 구조 확인 중...")
        validation_results['details']['file_structure'] = check_file_structure()
        
        # 4. 외부 서비스 확인
        print("   🔧 외부 서비스 확인 중...")
        validation_results['details']['services'] = check_services()
        
        # 5. 설정 파일 검증
        print("   ⚙️ 설정 파일 검증 중...")
        try:
            from ..config import settings
            config_dict = {
                'project': {},  # settings.project 제거
                'server': settings.server.dict() if hasattr(settings, 'server') else {},
                'llm': settings.llm.dict() if hasattr(settings, 'llm') else {}
            }
            validation_results['details']['config_validation'] = validate_config(config_dict)
        except Exception as e:
            validation_results['details']['config_validation'] = {
                'valid': False,
                'errors': [f"설정 로드 실패: {str(e)}"]
            }
        
        # 6. 모델 파일 확인
        print("   🤖 AI 모델 확인 중...")
        model_status = {'main_model': False, 'embedding_model': False}
        
        try:
            from ..config import settings
            if hasattr(settings, 'llm') and settings.llm:
                if hasattr(settings.llm, 'main_model'):
                    model_status['main_model'] = validate_model_path(settings.llm.main_model.path)
                if hasattr(settings.llm, 'embedding_model'):
                    model_status['embedding_model'] = validate_model_path(settings.llm.embedding_model.path)
        except Exception:
            pass
        
        validation_results['details']['models'] = model_status
        
        # 7. 전체 상태 결정
        status_scores = []
        
        # 시스템 요구사항 점수
        sys_req = validation_results['details']['system_requirements']
        critical_reqs = ['python_version', 'memory', 'disk_space']
        sys_score = sum(1 for req in critical_reqs 
                       if sys_req[req]['status'] in ['ok', 'excellent']) / len(critical_reqs)
        status_scores.append(('system', sys_score))
        
        # 의존성 점수
        deps = validation_results['details']['dependencies']
        if deps['status'] == 'ok':
            dep_score = 1.0
        elif deps['status'] == 'warning':
            dep_score = 0.7
        else:
            dep_score = 0.3
        status_scores.append(('dependencies', dep_score))
        
        # 파일 구조 점수
        files = validation_results['details']['file_structure']
        if files['status'] == 'ok':
            file_score = 1.0
        elif files['status'] == 'warning':
            file_score = 0.8
        else:
            file_score = 0.4
        status_scores.append(('files', file_score))
        
        # 전체 점수 계산
        overall_score = sum(score for _, score in status_scores) / len(status_scores)
        
        if overall_score >= 0.9:
            validation_results['overall_status'] = 'healthy'
        elif overall_score >= 0.7:
            validation_results['overall_status'] = 'caution'
        elif overall_score >= 0.5:
            validation_results['overall_status'] = 'warning'
        else:
            validation_results['overall_status'] = 'critical'
        
        # 8. 권장사항 생성
        recommendations = []
        
        # 시스템 권장사항
        if sys_req['memory']['status'] == 'warning':
            recommendations.append("메모리를 16GB 이상으로 업그레이드하는 것을 권장합니다.")
        
        if sys_req['gpu']['status'] == 'warning':
            recommendations.append("GPU를 사용하면 성능이 크게 향상됩니다.")
        
        # 의존성 권장사항
        if deps['critical']['missing']:
            recommendations.append(f"필수 패키지 설치: pip install {' '.join(deps['critical']['missing'])}")
        
        if deps['important']['missing']:
            recommendations.append(f"중요 패키지 설치: pip install {' '.join(deps['important']['missing'])}")
        
        # 모델 권장사항
        if not model_status['main_model']:
            recommendations.append("AI 모델 다운로드: python scripts/download_models.py")
        
        # 서비스 권장사항
        services = validation_results['details']['services']
        if services['docker']['status'] == 'not_installed':
            recommendations.append("Docker 설치를 권장합니다 (Neo4j 사용을 위해)")
        
        if services['neo4j']['status'] == 'not_running':
            recommendations.append("Neo4j 시작: docker-compose up -d neo4j")
        
        validation_results['recommendations'] = recommendations
        
        # 9. 요약 정보 생성
        validation_results['summary'] = {
            'overall_score': round(overall_score * 100, 1),
            'critical_issues': sum(1 for _, score in status_scores if score < 0.5),
            'warnings': sum(1 for _, score in status_scores if 0.5 <= score < 0.8),
            'recommendations_count': len(recommendations),
            'models_ready': sum(model_status.values()),
            'services_running': sum(1 for service in services.values() if service['status'] == 'ok')
        }
        
        validation_results['execution_time'] = round(time.time() - start_time, 2)
        
        print(f"✅ 검증 완료 ({validation_results['execution_time']}초)")
        
    except Exception as e:
        validation_results['overall_status'] = 'error'
        validation_results['details']['validation_error'] = str(e)
        validation_results['execution_time'] = round(time.time() - start_time, 2)
        print(f"❌ 검증 중 오류 발생: {e}")
    
    return validation_results

def generate_validation_report(results: Dict[str, Any]) -> str:
    """검증 결과 리포트 생성"""
    
    def status_icon(status: str) -> str:
        icons = {
            'ok': '✅', 'excellent': '🌟', 'warning': '⚠️', 
            'error': '❌', 'critical': '🚨', 'unknown': '❓',
            'not_installed': '📋', 'not_running': '💤',
            'healthy': '🟢', 'caution': '🟡'
        }
        return icons.get(status, '❓')
    
    report = []
    report.append("=" * 80)
    report.append("🔍 OPEN CODEAI 시스템 검증 리포트")
    report.append("=" * 80)
    
    # 기본 정보
    report.append(f"📅 검증 시각: {results['timestamp']}")
    report.append(f"⏱️  실행 시간: {results['execution_time']}초")
    report.append(f"🎯 전체 상태: {status_icon(results['overall_status'])} {results['overall_status'].upper()}")
    
    if 'summary' in results:
        summary = results['summary']
        report.append(f"📊 전체 점수: {summary['overall_score']}%")
        report.append(f"🚨 심각한 문제: {summary['critical_issues']}개")
        report.append(f"⚠️  경고: {summary['warnings']}개")
        report.append(f"🤖 준비된 모델: {summary['models_ready']}/2개")
        report.append(f"🔧 실행 중인 서비스: {summary['services_running']}개")
    
    report.append("")
    
    # 시스템 요구사항
    if 'system_requirements' in results['details']:
        report.append("💻 시스템 요구사항")
        report.append("-" * 40)
        
        sys_req = results['details']['system_requirements']
        
        report.append(f"🐍 Python: {status_icon(sys_req['python_version']['status'])} "
                     f"{sys_req['python_version']['current']} (요구: {sys_req['python_version']['required']})")
        
        report.append(f"💾 메모리: {status_icon(sys_req['memory']['status'])} "
                     f"{sys_req['memory']['current_gb']}GB (권장: {sys_req['memory']['recommended_gb']}GB)")
        
        report.append(f"💿 디스크: {status_icon(sys_req['disk_space']['status'])} "
                     f"{sys_req['disk_space']['current_gb']}GB (권장: {sys_req['disk_space']['recommended_gb']}GB)")
        
        report.append(f"🖥️  CPU: {status_icon(sys_req['cpu_cores']['status'])} "
                     f"{sys_req['cpu_cores']['current']}코어 (권장: {sys_req['cpu_cores']['recommended']}코어)")
        
        gpu_info = sys_req['gpu']
        gpu_text = f"🎮 GPU: {status_icon(gpu_info['status'])}"
        if gpu_info['available']:
            gpu_names = [dev['name'] for dev in gpu_info['devices']]
            gpu_text += f" {', '.join(gpu_names)}"
        else:
            gpu_text += " 사용 불가"
        report.append(gpu_text)
        
        report.append("")
    
    # 의존성 패키지
    if 'dependencies' in results['details']:
        report.append("📦 의존성 패키지")
        report.append("-" * 40)
        
        deps = results['details']['dependencies']
        categories = ['critical', 'important', 'optional']
        
        for category in categories:
            if category in deps:
                cat_data = deps[category]
                installed_count = len(cat_data['installed'])
                missing_count = len(cat_data['missing'])
                total = installed_count + missing_count
                
                category_icon = '🔴' if missing_count > 0 and category == 'critical' else \
                               '🟡' if missing_count > 0 and category == 'important' else '🟢'
                
                report.append(f"{category_icon} {category.title()}: {installed_count}/{total} 설치됨")
                
                if cat_data['missing']:
                    report.append(f"   ❌ 누락: {', '.join(cat_data['missing'])}")
        
        report.append("")
    
    # 파일 구조
    if 'file_structure' in results['details']:
        report.append("📁 파일 구조")
        report.append("-" * 40)
        
        files = results['details']['file_structure']
        report.append(f"{status_icon(files['status'])} 전체 상태: {files['status']}")
        
        if files['directories']['missing']:
            report.append(f"❌ 누락된 디렉토리: {', '.join(files['directories']['missing'])}")
        
        if files['files']['missing']:
            report.append(f"❌ 누락된 파일: {', '.join(files['files']['missing'])}")
        
        report.append("")
    
    # 외부 서비스
    if 'services' in results['details']:
        report.append("🔧 외부 서비스")
        report.append("-" * 40)
        
        services = results['details']['services']
        
        for service_name, service_data in services.items():
            status = service_data['status']
            report.append(f"{status_icon(status)} {service_name.title()}: {status}")
            
            if service_name == 'docker' and service_data.get('version'):
                report.append(f"   📋 버전: {service_data['version']}")
        
        report.append("")
    
    # AI 모델
    if 'models' in results['details']:
        report.append("🤖 AI 모델")
        report.append("-" * 40)
        
        models = results['details']['models']
        
        for model_name, is_ready in models.items():
            icon = '✅' if is_ready else '❌'
            status = '준비됨' if is_ready else '누락'
            report.append(f"{icon} {model_name}: {status}")
        
        report.append("")
    
    # 설정 검증
    if 'config_validation' in results['details']:
        report.append("⚙️ 설정 검증")
        report.append("-" * 40)
        
        config = results['details']['config_validation']
        
        if config.get('valid', False):
            report.append("✅ 설정 파일이 유효합니다")
        else:
            report.append("❌ 설정 파일에 문제가 있습니다")
        
        if config.get('errors'):
            report.append("🚨 오류:")
            for error in config['errors']:
                report.append(f"   • {error}")
        
        if config.get('warnings'):
            report.append("⚠️ 경고:")
            for warning in config['warnings']:
                report.append(f"   • {warning}")
        
        report.append("")
    
    # 권장사항
    if results.get('recommendations'):
        report.append("💡 권장사항")
        report.append("-" * 40)
        
        for i, recommendation in enumerate(results['recommendations'], 1):
            report.append(f"{i}. {recommendation}")
        
        report.append("")
    
    # 다음 단계
    report.append("🚀 다음 단계")
    report.append("-" * 40)
    
    status = results['overall_status']
    
    if status == 'healthy':
        report.append("🎉 시스템이 정상적으로 구성되었습니다!")
        report.append("   다음 명령으로 서버를 시작할 수 있습니다:")
        report.append("   ./start.sh")
        report.append("")
        report.append("   Continue.dev 설정:")
        report.append("   1. VS Code에서 Continue 확장 설치")
        report.append("   2. 설정 파일 복사: cp examples/continue_config.json ~/.continue/config.json")
        report.append("   3. Ctrl+Shift+L로 채팅 시작")
    
    elif status == 'caution':
        report.append("⚠️ 시스템이 대체로 정상이지만 일부 개선이 필요합니다.")
        report.append("   위의 권장사항을 확인하여 최적화할 수 있습니다.")
        report.append("   기본 기능은 사용 가능합니다: ./start.sh")
    
    elif status == 'warning':
        report.append("🔧 여러 문제가 발견되었습니다. 다음 순서로 해결하세요:")
        report.append("   1. 필수 패키지 설치: pip install -r requirements.txt")
        report.append("   2. 모델 다운로드: python scripts/download_models.py")
        report.append("   3. Docker 서비스 시작: docker-compose up -d")
        report.append("   4. 재검증: python scripts/system_check.py --detailed")
    
    else:  # critical or error
        report.append("🚨 심각한 문제가 발견되었습니다. 즉시 해결이 필요합니다:")
        report.append("   1. Python 가상환경 확인: source venv/bin/activate")
        report.append("   2. 시스템 요구사항 확인 (메모리, 디스크 공간)")
        report.append("   3. 설치 가이드 참조: README.md")
        report.append("   4. 문제 지속시 이슈 생성: https://github.com/ChangooLee/open-codeai/issues")
    
    report.append("")
    report.append("=" * 80)
    report.append("검증 완료")
    report.append("=" * 80)
    
    return "\n".join(report)