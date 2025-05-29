"""
하드웨어 정보 및 성능 감지 유틸리티
"""
import os
import platform
from typing import Dict, Any, Optional, List
import psutil


def get_system_info() -> Dict[str, Any]:
    """시스템 기본 정보 수집"""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }


def get_memory_info() -> Dict[str, Any]:
    """메모리 정보 수집"""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_gb": round(memory.used / (1024**3), 2),
        "percent": memory.percent,
        "swap_total_gb": round(swap.total / (1024**3), 2),
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "swap_percent": swap.percent
    }


def check_gpu_availability() -> Dict[str, Any]:
    """GPU 사용 가능성 및 정보 확인"""
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "cuda_version": None,
        "driver_version": None
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            
            # 각 GPU 정보 수집
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "id": i,
                    "name": device_props.name,
                    "memory_gb": round(device_props.total_memory / (1024**3), 2),
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                })
    except ImportError:
        pass
    
    # NVIDIA-SMI로 추가 정보 수집 시도
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = line.split(', ')
                if len(parts) >= 3 and i < len(gpu_info["devices"]):
                    gpu_info["devices"][i]["driver_version"] = parts[2]
                    if not gpu_info["driver_version"]:
                        gpu_info["driver_version"] = parts[2]
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return gpu_info


def get_disk_info() -> Dict[str, Any]:
    """디스크 정보 수집"""
    disk_usage = psutil.disk_usage('/')
    
    return {
        "total_gb": round(disk_usage.total / (1024**3), 2),
        "used_gb": round(disk_usage.used / (1024**3), 2),
        "free_gb": round(disk_usage.free / (1024**3), 2),
        "percent": round((disk_usage.used / disk_usage.total) * 100, 2)
    }


def get_network_info() -> Dict[str, Any]:
    """네트워크 정보 수집"""
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        return {
            "hostname": hostname,
            "local_ip": local_ip,
            "network_interfaces": len(psutil.net_if_addrs())
        }
    except Exception:
        return {"hostname": "unknown", "local_ip": "unknown", "network_interfaces": 0}


def get_hardware_info() -> Dict[str, Any]:
    """전체 하드웨어 정보 수집"""
    return {
        "system": get_system_info(),
        "memory": get_memory_info(),
        "gpu": check_gpu_availability(),
        "disk": get_disk_info(),
        "network": get_network_info(),
        "timestamp": psutil.boot_time()
    }


def recommend_settings(hardware_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """하드웨어 사양에 따른 권장 설정"""
    if hardware_info is None:
        hardware_info = get_hardware_info()
    
    memory_gb = hardware_info["memory"]["total_gb"]
    cpu_count = hardware_info["system"]["cpu_count"]
    gpu_available = hardware_info["gpu"]["available"]
    gpu_memory = 0
    
    if gpu_available and hardware_info["gpu"]["devices"]:
        gpu_memory = max(device["memory_gb"] for device in hardware_info["gpu"]["devices"])
    
    # 메모리 기반 권장 설정
    if memory_gb >= 128:
        workers = min(32, cpu_count * 2)
        cache_size = 20
        memory_limit = 64
    elif memory_gb >= 64:
        workers = min(16, cpu_count)
        cache_size = 10
        memory_limit = 32
    elif memory_gb >= 32:
        workers = min(8, cpu_count)
        cache_size = 5
        memory_limit = 16
    else:
        workers = min(4, cpu_count)
        cache_size = 2
        memory_limit = 8
    
    # GPU 기반 권장 설정
    gpu_settings = {
        "enable": gpu_available,
        "memory_fraction": 0.8 if gpu_memory >= 16 else 0.7,
        "mixed_precision": gpu_available and gpu_memory >= 8
    }
    
    return {
        "parallel_workers": workers,
        "cache_size_gb": cache_size,
        "memory_limit_gb": memory_limit,
        "gpu": gpu_settings,
        "batch_size": 64 if memory_gb >= 32 else 32
    }


def check_requirements() -> Dict[str, bool]:
    """시스템 요구사항 체크"""
    hardware_info = get_hardware_info()
    memory_gb = hardware_info["memory"]["total_gb"]
    disk_gb = hardware_info["disk"]["free_gb"]
    
    requirements = {
        "memory_16gb": memory_gb >= 16,
        "memory_32gb": memory_gb >= 32,  # 권장
        "memory_64gb": memory_gb >= 64,  # 최적
        "disk_50gb": disk_gb >= 50,
        "disk_200gb": disk_gb >= 200,    # 권장
        "gpu_available": hardware_info["gpu"]["available"],
        "gpu_12gb": False,
        "gpu_16gb": False,
        "cpu_8cores": hardware_info["system"]["cpu_count"] >= 8,
        "cpu_16cores": hardware_info["system"]["cpu_count"] >= 16
    }
    
    # GPU 메모리 체크
    if hardware_info["gpu"]["available"] and hardware_info["gpu"]["devices"]:
        max_gpu_memory = max(device["memory_gb"] for device in hardware_info["gpu"]["devices"])
        requirements["gpu_12gb"] = max_gpu_memory >= 12
        requirements["gpu_16gb"] = max_gpu_memory >= 16
    
    return requirements


def get_performance_profile() -> str:
    """하드웨어 기반 성능 프로필 결정"""
    requirements = check_requirements()
    
    if requirements["memory_64gb"] and requirements["gpu_16gb"] and requirements["cpu_16cores"]:
        return "enterprise"
    elif requirements["memory_32gb"] and requirements["gpu_12gb"] and requirements["cpu_8cores"]:
        return "professional"
    elif requirements["memory_16gb"] and requirements["cpu_8cores"]:
        return "standard"
    else:
        return "minimal"


if __name__ == "__main__":
    # 테스트 실행
    import json
    hardware_info = get_hardware_info()
    print("=== Hardware Information ===")
    print(json.dumps(hardware_info, indent=2))
    
    print("\n=== Recommended Settings ===")
    recommended = recommend_settings(hardware_info)
    print(json.dumps(recommended, indent=2))
    
    print(f"\n=== Performance Profile ===")
    print(f"Profile: {get_performance_profile()}")
    
    print(f"\n=== Requirements Check ===")
    requirements = check_requirements()
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {req}: {status}")