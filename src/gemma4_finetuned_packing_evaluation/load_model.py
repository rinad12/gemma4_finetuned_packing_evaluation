import os
from enum import Enum

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, QuantoConfig

BASE_MODEL_ID = "google/gemma-4-E4B"
ADAPTER_ID = "rinad12/gemma_4_fine_tuned_packing"


class CpuQuantization(str, Enum):
    int8 = "int8"
    int4 = "int4"


def load_model(
    load_in_4bit: bool = False,
    cpu_quantize: CpuQuantization = CpuQuantization.int4,
):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Add it to your .env file: HF_TOKEN=hf_xxx")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, token=token)

    quantization_config = None
    if load_in_4bit:
        if not torch.cuda.is_available():
            print("Warning: 4-bit quantization requires CUDA. Falling back to --cpu_quantize int8.")
            cpu_quantize = "int8"
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    if cpu_quantize and not torch.cuda.is_available():
        # INT8 ~5 GB, INT4 ~2.5 GB — works on CPU without CUDA
        quantization_config = QuantoConfig(weights=cpu_quantize)

    dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    # QuantoConfig (CPU) is incompatible with device_map
    device_map = None if quantization_config and not torch.cuda.is_available() else "auto"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        dtype=dtype,
        device_map=device_map,
        token=token,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=token)

    model.eval()
    return model, tokenizer
