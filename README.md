# infra — SageMaker 모델 배포

> **운영용 모델 배포 코드.** Qwen2.5 답변·라우터·메모리 모델을 AWS SageMaker 엔드포인트로 배포합니다. 오케스트레이터(`LLM_BACKEND=sagemaker`)가 이 엔드포인트들을 호출합니다.

전체 프로젝트 개요는 대표 저장소 [docs](https://github.com/2026-Pretrained-Conversational-Model/docs)를 참고하세요.

---

## 구성

```
deploy/
├── deploy_all.py       3개 모델을 SageMaker 엔드포인트로 일괄 배포
└── sagemaker_code/
    ├── inference.py    커스텀 추론 핸들러(model_fn / input_fn / predict_fn)
    └── requirements.txt
```

## 배포되는 엔드포인트

| 엔드포인트 | 모델 | 역할 |
| --- | --- | --- |
| `ai-orchestrator-answer-v2` | Qwen/Qwen2.5-7B-Instruct | 답변 생성 |
| `ai-orchestrator-router-v2` | Qwen/Qwen2.5-3B-Instruct | RAG 라우팅 |
| `ai-orchestrator-memory-v2` | qwen2.5-3b-memory-summary (파인튜닝) | 메모리 요약 |

모든 모델은 `USE_4BIT=true`(NF4)로 로딩됩니다.

## 추론 계약 (inference.py)

오케스트레이터의 SageMaker 백엔드와 정확히 맞춰진 입출력 형식입니다.

```
요청  : {"system": str, "user": str, "max_new_tokens": int}   (application/json)
응답  : {"text": str}
```

`predict_fn`은 system/user를 chat template으로 조립해 생성하고, 새로 생성된 토큰만 디코딩해 `text`로 반환합니다.

## 배포

```bash
pip install -r deploy/sagemaker_code/requirements.txt
python deploy/deploy_all.py     # AWS 자격증명 필요(역할/키)
# 배포 후 AWS 콘솔에서 엔드포인트 상태 확인
```

> **정리 필요 메모(포폴):**
> - `deploy_all.py`에 머신 고정 절대경로(`/Users/.../infra/.venv/...`)가 하드코딩되어 있어 제거 권장.
> - 메모리 모델 id가 배포 스크립트(`g34634/...`)와 실제 학습본(`yeseul0-0/...v0.3`)으로 갈려 있어 통일 권장.
> - git 저장소가 `infra/infra/`에 중첩되어 있어 한 단계 평탄화 권장.

---

## 담당 역할

- SageMaker 배포·인프라: **정찬희**
