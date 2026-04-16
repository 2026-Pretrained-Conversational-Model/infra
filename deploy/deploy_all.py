# deploy/deploy_all.py
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel

ROLE_ARN = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/<YOUR_SAGEMAKER_ROLE>"
REGION = "ap-northeast-2"

session = sagemaker.Session()

SOURCE_DIR = "sagemaker_code"
ENTRY_POINT = "inference.py"

COMMON = {
    "role": ROLE_ARN,
    "entry_point": ENTRY_POINT,
    "source_dir": SOURCE_DIR,
    "transformers_version": "4.44.2",
    "pytorch_version": "2.4.0",
    "py_version": "py311",
}

MODELS = [
    {
        "name": "ai-orchestrator-answer-v1",
        "env": {
            "MODEL_ID": "Qwen/Qwen2.5-7B-Instruct",
            "TOKENIZER_ID": "Qwen/Qwen2.5-7B-Instruct",
            "USE_4BIT": "true",
            "DEFAULT_MAX_NEW_TOKENS": "200",
            "TEMPERATURE": "0.0",
            "DO_SAMPLE": "false",
        },
        "instance_type": "ml.g5.12xlarge",
    },
    {
        "name": "ai-orchestrator-router-v1",
        "env": {
            "MODEL_ID": "Qwen/Qwen2.5-3B-Instruct",
            "TOKENIZER_ID": "Qwen/Qwen2.5-3B-Instruct",
            "USE_4BIT": "true",
            "DEFAULT_MAX_NEW_TOKENS": "32",
            "TEMPERATURE": "0.0",
            "DO_SAMPLE": "false",
        },
        "instance_type": "ml.g5.2xlarge",
    },
    {
        "name": "ai-orchestrator-summary-v1",
        "env": {
            "MODEL_ID": "g34634/qwen2.5-3b-memory-summary-v1",
            "TOKENIZER_ID": "Qwen/Qwen2.5-3B-Instruct",
            "USE_4BIT": "true",
            "DEFAULT_MAX_NEW_TOKENS": "300",
            "TEMPERATURE": "0.0",
            "DO_SAMPLE": "false",
        },
        "instance_type": "ml.g5.2xlarge",
    },
]

for spec in MODELS:
    print(f"Deploying {spec['name']} ...")

    model = HuggingFaceModel(
        env=spec["env"],
        **COMMON,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=spec["instance_type"],
        endpoint_name=spec["name"],
    )

    print(f"Done: {predictor.endpoint_name}")