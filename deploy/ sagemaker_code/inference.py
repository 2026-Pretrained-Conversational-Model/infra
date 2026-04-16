# deploy/sagemaker_code/inference.py
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = os.environ["MODEL_ID"]
TOKENIZER_ID = os.environ.get("TOKENIZER_ID", MODEL_ID)
USE_4BIT = os.environ.get("USE_4BIT", "false").lower() == "false"
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "800"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "false").lower() == "true"
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))

def model_fn(model_dir):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id

    model.eval()
    return {"model": model, "tokenizer": tok}

def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    return json.loads(request_body)

def predict_fn(data, artifacts):
    model = artifacts["model"]
    tok = artifacts["tokenizer"]

    system = data.get("system", "") or ""
    user = data.get("user", "") or ""
    max_new_tokens = int(data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))

    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    prompt = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tok([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(generated_ids, skip_special_tokens=True).strip()

    return {"text": text}

def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction, ensure_ascii=False), accept