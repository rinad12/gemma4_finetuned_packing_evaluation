"""
Evaluation script for QLoRA fine-tuned Gemma 4 model.

Usage:
    uv run main.py --prompt "Your prompt here"
"""

import argparse

import torch

from gemma4_finetuned_packing_evaluation.load_model import CpuQuantization, load_model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a fine-tuned Gemma 4 model")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit (requires CUDA)")
    parser.add_argument("--cpu_quantize", choices=[e.value for e in CpuQuantization], default=CpuQuantization.int4.value, help="CPU quantization: int8 (~5 GB) or int4 (~2.5 GB, default)")
    args = parser.parse_args()

    cpu_quantize = CpuQuantization(args.cpu_quantize)
    model, tokenizer = load_model(args.load_in_4bit, cpu_quantize)

    print(f"\nPrompt: {args.prompt}")
    response = generate(model, tokenizer, args.prompt, args.max_new_tokens)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
