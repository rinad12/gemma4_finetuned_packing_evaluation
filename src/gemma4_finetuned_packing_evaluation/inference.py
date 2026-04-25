import time

import torch

from .config import MAX_NEW_TOKENS, SYSTEM_PROMPT


def generate_for_eval(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    few_shot_pairs: list[tuple[str, str]] | None = None,
) -> tuple[str, float, int]:
    """Returns (decoded_text, tokens_per_second, new_token_count)."""
    def wrap(text: str) -> list[dict]:
        return [{"type": "text", "text": text}]

    messages = []
    base_content = f"{SYSTEM_PROMPT}\n\n"

    if few_shot_pairs:
        for i, (user_ex, asst_ex) in enumerate(few_shot_pairs):
            u_text = (base_content + user_ex) if i == 0 else user_ex
            messages.append({"role": "user",      "content": wrap(u_text)})
            messages.append({"role": "assistant", "content": wrap(asst_ex)})
        messages.append({"role": "user", "content": wrap(prompt)})
    else:
        messages.append({"role": "user", "content": wrap(base_content + prompt)})

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text=formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.05,
            no_repeat_ngram_size=6,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = output_ids.shape[1] - prompt_len
    tps = round(new_tokens / elapsed, 2) if elapsed > 0 else 0.0
    decoded = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return decoded, tps, new_tokens
