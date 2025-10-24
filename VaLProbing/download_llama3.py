# This source code is licensed under the MIT license.

import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm

# -------------------------------
# Load dataset
# -------------------------------
dataset = load_dataset("In2Training/VaLProbing-32K")
categories = list(dataset.keys())

# -------------------------------
# Create output directory
# -------------------------------
output_dir = "./ValProbing-32K/plaintext/"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Utility: clean text
# -------------------------------
def clean_prompt(text: str) -> str:
    """Remove Llama-style control tokens and extra whitespace."""
    text = re.sub(r"\[/?INST\]", "", text)
    text = re.sub(r"<<SYS>>|<</SYS>>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# Iterate over categories
# -------------------------------
for cate in categories:
    out_path = os.path.join(output_dir, f"{cate}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f_out:
        for info in tqdm(dataset[cate], desc=f"Processing {cate}"):
            # Remove empty-string fields
            for key in list(info.keys()):
                if isinstance(info[key], str) and info[key].strip() == "":
                    info.pop(key)

            # Extract and clean prompt / completion
            prompt = clean_prompt(info.get("prompt", ""))
            completion = info.get("completion", "").strip()

            # Create plain, model-agnostic entry
            clean_entry = {
                "system": "You are a helpful assistant.",
                "user": prompt,
                "assistant": completion,
                "meta": {
                    "set_id": info.get("set_id"),
                    "position_id": info.get("position_id"),
                    "category": cate
                }
            }

            # Write as JSONL
            f_out.write(json.dumps(clean_entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved {cate} → {out_path}")

print("🎯 Conversion complete! All plaintext JSONL files saved in ./ValProbing-32K/plaintext/")