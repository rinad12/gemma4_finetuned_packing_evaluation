"""
Evaluation script for QLoRA fine-tuned Gemma 4 model.

Usage:
    uv run main.py --model_id <hf-repo-id> --prompt "Your prompt here"
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_id: str, base_model_id: str | None = None, load_in_4bit: bool = False):
    """Load fine-tuned model from HuggingFace Hub.

    If base_model_id is provided, loads as a PEFT adapter on top of the base model.
    Otherwise loads the merged model directly.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = None
    if load_in_4bit:
        # 4-bit quantization requires CUDA — skip on CPU
        if not torch.cuda.is_available():
            print("Warning: 4-bit quantization requires CUDA. Loading in float32 instead.")
            load_in_4bit = False
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    if base_model_id:
        # Load base model + PEFT adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        # Load merged/standalone fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float32,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned Gemma 4 model")
    parser.add_argument("--model_id", required=True, help="HuggingFace repo ID of the fine-tuned model")
    parser.add_argument("--base_model_id", default=None, help="HuggingFace repo ID of the base model (if model_id is a PEFT adapter)")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit (requires CUDA)")
    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    model, tokenizer = load_model(args.model_id, args.base_model_id, args.load_in_4bit)

    print(f"\nPrompt: {args.prompt}")
    response = generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
