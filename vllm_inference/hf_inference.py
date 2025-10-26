# This source code is licensed under the MIT license

import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Helper: Automatically apply chat template if available
# ============================================================
def format_prompt(tokenizer, prompt):
    """
    Wrap a user prompt in the model's chat template if available.
    Works for chat-tuned models like Llama-2-Chat, Llama-3, Mistral-Instruct, etc.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        conversation = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[WARN] Failed to apply chat template: {e}")
            return prompt
    else:
        # Base (non-chat) model: return plain text
        return prompt


# ============================================================
# Main inference function
# ============================================================
def inference(
    testdata_folder,
    testdata_file,
    output_folder,
    output_file,
    model_path,
    max_length,
    trust_remote_code,
    batch_size,
    input_max_length,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # === Load dataset ===
    input_path = os.path.join(testdata_folder, testdata_file)
    with open(input_path, "r", encoding="utf-8") as f_read:
        dataset = [json.loads(line) for line in f_read.readlines()]

    test_prompts = []
    for ex in dataset:
        system_msg = ex.get("system", "")
        user_msg = ex.get("user", "")

        if system_msg:
            prompt = f"{system_msg}\n{user_msg}"
        else:
            prompt = user_msg

        test_prompts.append(prompt)

    total_lines = len(test_prompts)
    logger.info(f"Loaded {total_lines} prompts from {input_path}")
    assert total_lines > 0

    # === Load model & tokenizer ===
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    temperature = 0.0
    top_p = 1.0
    max_new_tokens = max_length
    os.makedirs(output_folder, exist_ok=True)
    all_outputs = []

    for i in tqdm(range(0, total_lines, batch_size), desc="Generating"):
        batch_prompts = [format_prompt(tokenizer, p) for p in test_prompts[i:i + batch_size]]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=input_max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        for text in decoded:
            all_outputs.append({"samples": [text.strip()]})

        logger.info(f"Processed {min(i + batch_size, total_lines)} / {total_lines}")

    # === Save results ===
    output_path = os.path.join(output_folder, output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for output in all_outputs:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    logger.info(f"✅ Saved predictions to {output_path}")


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace model inference script")
    parser.add_argument("--testdata_folder", type=str, required=True)
    parser.add_argument("--testdata_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--trust_remote_code", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_max_length", type=int, default=32768)
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = f"sample_{args.testdata_file}"

    os.makedirs(args.output_folder, exist_ok=True)

    inference(
        testdata_folder=args.testdata_folder,
        testdata_file=args.testdata_file,
        output_folder=args.output_folder,
        output_file=args.output_file,
        model_path=args.model_path,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        batch_size=args.batch_size,
        input_max_length=args.input_max_length,
    )