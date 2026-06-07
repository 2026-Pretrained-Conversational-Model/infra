import sys
import os

# 1. [경로 점령] 님의 .venv를 0순위로 강제 주입합니다. 
# 이 주소는 님의 터미널 로그에서 확인된 '주소'입니다.
true_venv_path = "/Users/gim-yeseul/Documents/2025_김예슬/2026-NLP-03/infra/.venv/lib/python3.9/site-packages"

if true_venv_path not in sys.path:
    sys.path.insert(0, true_venv_path)

# 2. [모듈 로드] 2.x 버전의 안정적인 SageMaker를 깨웁니다.
try:
    import sagemaker
    from sagemaker.huggingface.model import HuggingFaceModel
    print(f" [성공] 모든 억까를 뚫고 SageMaker 로드 완료! (버전: {sagemaker.__version__})")
    print(f" 현재 탐색 1순위: {sys.path[0]}")
except ImportError as e:
    print(f"[실패] 모듈을 찾을 수 없습니다: {e}")
    print("해결책: 'pip install --no-cache-dir \"sagemaker[all]<3.0.0\"'을 실행했는지 확인하세요.")
    sys.exit(1)

# 3. [배포 설정] AWS의 문을 열기 위한 설정값들입니다.
# 참고: 한국어(Korean) 서비스 지원을 위한 모델들입니다.
ROLE_ARN = "arn:aws:iam::622165781875:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

# 가격 안내: ml.g5.12xlarge 기준 시간당 약 $5.67 (한화 약 7,650원) 
MODELS = [
    {
        "name": "answer-v2", 
        "id": "Qwen/Qwen2.5-7B-Instruct", 
        "tok_id": "Qwen/Qwen2.5-7B-Instruct",
        "ins": "ml.g5.12xlarge"
    },
    {
        "name": "router-v2", 
        "id": "Qwen/Qwen2.5-3B-Instruct", 
        "tok_id": "Qwen/Qwen2.5-3B-Instruct",
        "ins": "ml.g5.2xlarge"
    },
    {
        "name": "memory-v2", 
        "id": "g34634/qwen2.5-3b-memory-summary-v1", 
        "tok_id": "Qwen/Qwen2.5-3B-Instruct",
        "ins": "ml.g5.2xlarge"
    },
]

# 4. [배포 실행] 루프를 돌며 AWS로 명령어를 송출합니다.
for m in MODELS:
    endpoint_name = f"ai-orchestrator-{m['name']}"
    print(f" [{m['name']}] 모델 배포를 시작합니다... (인스턴스: {m['ins']})")
    
    hfm = HuggingFaceModel(
        env={
            "MODEL_ID": m['id'], 
            "TOKENIZER_ID": m['tok_id'], 
            "USE_4BIT": "true" 
        }, 
        role=ROLE_ARN, 
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version="py310",
        source_dir="sagemaker_code",
        entry_point="inference.py"
    )
    
    # 실제 배포 명령 (이 코드가 실행되면 AWS 비용이 발생합니다) 

    hfm.deploy(
        initial_instance_count=1, 
        instance_type=m['ins'], 
        endpoint_name=endpoint_name
    )
    print(f" [{m['name']}] 배포 명령 전송 완료! (AWS 콘솔을 확인하세요)")

print("\n 모든 모델 배포 명령이 성공적으로 전송되었습니다. ")