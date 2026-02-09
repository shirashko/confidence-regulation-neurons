import torch
import os
import json
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()


def load_balanced_certainty_data(n_samples=150):
    """
    Collects high-confidence (certain) samples from PopQA and AmbigQA.
    """
    baseline_data = []

    # A. PopQA - Filter for highly popular entities
    print("Loading PopQA...")
    popqa = load_dataset("akariasai/PopQA", split="test")
    # Using 50,000 threshold to ensure model familiarity
    popular_samples = popqa.filter(lambda x: x['s_pop'] > 50000)

    for i in range(min(n_samples, len(popular_samples))):
        baseline_data.append({
            "question": popular_samples[i]['question'],
            "ground_truth": popular_samples[i]['possible_answers'],
            "source": "PopQA"
        })

    # B. AmbigQA - Filter for unambiguous questions
    print("Loading AmbigQA...")
    ambigqa = load_dataset("sewon/ambig_qa", "light", split="train")

    def is_single_answer(example):
        try:
            # Check if the annotation type is explicitly 'single_answer'
            anno_types = example['annotations']['type']
            return 'single_answer' in anno_types or anno_types[0] == 'single_answer'
        except (KeyError, IndexError):
            return False

    single_answer_samples = ambigqa.filter(is_single_answer)

    # Fallback mechanism if the filter is too strict for specific dataset versions
    if len(single_answer_samples) == 0:
        print("Warning: Filter failed. Falling back to first available samples.")
        single_answer_samples = ambigqa.select(range(n_samples))

    print(f"Found {len(single_answer_samples)} single answer samples.")

    for i in range(min(n_samples, len(single_answer_samples))):
        baseline_data.append({
            "question": single_answer_samples[i]['question'],
            "ground_truth": single_answer_samples[i]['annotations']['answer'][0],
            "source": "AmbigQA"
        })

    return baseline_data


def compute_verified_baseline(data_items):
    """
    Computes the mean activation vector (x_base) while verifying model confidence.
    """
    activations = []
    verification_stats = []
    used_prompts = []  # List to track the exact inputs used

    print(f"Computing activations for {len(data_items)} samples...")
    for item in tqdm(data_items):
        # Using a simple Q&A format that GPT-2 Small handles well
        prompt = f"Question: {item['question']}\nAnswer:"

        # Save prompt metadata for reproducibility
        used_prompts.append({
            "prompt": prompt,
            "source": item["source"],
            "ground_truth": item["ground_truth"]
        })

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Extract confidence (softmax probability of the predicted token)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_prob, top_idx = torch.max(probs, dim=0)
            predicted_token = tokenizer.decode(top_idx).strip()

            # Extract the Residual Stream activation from the final layer
            # For GPT-2, hidden_states[-1] is the output of the final transformer block
            last_token_act = outputs.hidden_states[-1][0, -1, :]

            activations.append(last_token_act.cpu())
            verification_stats.append({
                "source": item["source"],
                "prediction": predicted_token,
                "confidence": top_prob.item()
            })

    # Calculate the centroid (the mean vector x_base)
    x_base = torch.stack(activations).mean(dim=0)

    # Save the input logs to JSON
    with open("data/baseline_inputs.json", "w", encoding="utf-8") as f:
        json.dump(used_prompts, f, indent=4, ensure_ascii=False)

    stats_df = pd.DataFrame(verification_stats)
    print(f"\nAverage Baseline Confidence: {stats_df['confidence'].mean():.4f}")
    return x_base


# Main Execution
os.makedirs("data", exist_ok=True)
certainty_data = load_balanced_certainty_data(n_samples=150)
x_base = compute_verified_baseline(certainty_data)

# Save the final baseline vector
torch.save(x_base, "data/common_certainty_baseline.pt")

print(f"Baseline vector saved to data/common_certainty_baseline.pt")
print(f"Baseline inputs saved to data/baseline_inputs.json")