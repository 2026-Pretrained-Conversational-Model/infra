import sys
import os

# 1. ls로 확인한 실제 경로를 변수에 저장합니다.
# 점(.)이 붙은 .venv 경로를 0순위로 강제 주입합니다.
true_path = "/Users/chan/Documents/프로젝트2/.venv/lib/python3.9/site-packages"

if true_path not in sys.path:
    sys.path.insert(0, true_path)
    print(f"✅ [경로 강제 점령] {true_path}")

# 2. 임포트 시도
try:
    import sagemaker
    from sagemaker.huggingface import HuggingFaceModel
    print(f"✅ [성공] 드디어 모든 모듈을 찾았습니다! (SDK 버전: {sagemaker.__version__})")
except ImportError as e:
    print(f"❌ 여전히 못 찾음: {e}")
    # 현재 파이썬이 실제로 뒤지고 있는 리스트를 다 보여줘서 범인을 잡습니다.
    print(f" 현재 sys.path 전체 리스트: {sys.path[:3]}")
    sys.exit(1)


# --- 배포 로직 시작 ---
ROLE_ARN = "arn:aws:iam::622165781875:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
MODELS = [
    {"name": "answer-v1", "id": "Qwen/Qwen2.5-7B-Instruct", "tok_id": "Qwen/Qwen2.5-7B-Instruct", "ins": "ml.g5.12xlarge"},
    {"name": "router-v1", "id": "Qwen/Qwen2.5-3B-Instruct", "tok_id": "Qwen/Qwen2.5-3B-Instruct", "ins": "ml.g5.2xlarge"},
    {"name": "memory-v1", "id": "g34634/qwen2.5-3b-memory-summary-v1", "tok_id": "Qwen/Qwen2.5-3B-Instruct", "ins": "ml.g5.2xlarge"},
]

for m in MODELS:
    endpoint_name = f"ai-orchestrator-{m['name']}"
    print(f"🛰️ {m['name']} 배포 명령 전송 중...")
    hfm = HuggingFaceModel(
        env={"MODEL_ID": m['id'], "TOKENIZER_ID": m['tok_id'], "USE_4BIT": "true"}, 
        role=ROLE_ARN, 
        transformers_version="4.37.0",
        pytorch_version="2.1.0",
        py_version="py310",
        source_dir="sagemaker_code", 
        entry_point="inference.py"
    )
    hfm.deploy(initial_instance_count=1, instance_type=m['ins'], endpoint_name=endpoint_name)
    print(f"✅ {m['name']} 배포 완료!")
