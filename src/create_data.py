import torch
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()

# Prompt configuration
FEW_SHOT_BASELINE = "Question: What is the capital of France? Answer: Paris\n"
PROMPT_TEMPLATE = "Question: {}\nAnswer:"


# --- DATA LOADING ---

def get_baseline_data(n_samples=150):
    """Gathers high-popularity and unambiguous samples for the certainty baseline."""
    print("Loading PopQA (High Pop)...")
    popqa = load_dataset("akariasai/PopQA", split="test")
    popular = popqa.filter(lambda x: x['s_pop'] > 100000)
    popular = popular.select(range(min(n_samples, len(popular))))

    print("Loading AmbigQA (Filtering for Single Answer)...")
    ambigqa = load_dataset("sewon/ambig_qa", "light", split="train")

    def is_single(ex):
        try:
            return 'single_answer' in ex['annotations']['type']
        except:
            return False

    single = ambigqa.filter(is_single)
    if len(single) == 0:
        single = ambigqa.select(range(min(n_samples, len(ambigqa))))
    else:
        single = single.select(range(min(n_samples, len(single))))

    data = []
    for item in popular: data.append({"question": item['question'], "type": "baseline", "source": "PopQA"})
    for item in single: data.append({"question": item['question'], "type": "baseline", "source": "AmbigQA"})
    return data


def get_uncertain_data(n_samples=300):
    """Gathers the epistemic (unknown) and aleatoric (ambiguous) test samples."""
    print("Loading Uncertain data...")
    popqa = load_dataset("akariasai/PopQA", split="test")
    # Epistemic: low popularity
    epistemic = popqa.filter(lambda x: x['s_pop'] < 100).select(range(min(n_samples, len(popqa))))

    ambigqa = load_dataset("sewon/ambig_qa", "light", split="train")

    def is_ambig(ex):
        try:
            return 'multipleQAs' in ex['annotations']['type']
        except:
            return False

    # Aleatoric: multiple answers/interpretations
    aleatoric = ambigqa.filter(is_ambig).select(range(min(n_samples, len(ambigqa))))

    data = []
    for item in epistemic: data.append({"question": item['question'], "type": "epistemic", "label": 0})
    for item in aleatoric: data.append({"question": item['question'], "type": "aleatoric", "label": 1})
    return data


# --- PROCESSING ---

def process_and_compare():
    baseline_items = get_baseline_data()
    uncertain_items = get_uncertain_data()
    os.makedirs("data", exist_ok=True)

    activations_baseline = []
    verified_baseline_inputs = []  # Tracking only inputs that pass the certainty threshold
    conf_scores = {"baseline": [], "epistemic": [], "aleatoric": []}

    print("\nProcessing Baseline (Applying threshold filtering)...")
    for item in tqdm(baseline_items):
        prompt = FEW_SHOT_BASELINE + PROMPT_TEMPLATE.format(item['question'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            prob = torch.max(torch.softmax(outputs.logits[0, -1, :], dim=-1)).item()

            # GOLD STANDARD FILTER: Only include if GPT-2 is genuinely certain
            if prob > 0.20:
                activation = outputs.hidden_states[-1][0, -1, :].cpu()
                activations_baseline.append(activation)
                conf_scores["baseline"].append(prob)

                # Log only the samples that contribute to the mean vector
                verified_baseline_inputs.append({
                    "prompt": prompt,
                    "confidence": prob,
                    "source": item["source"]
                })

    print("Processing Uncertain Samples...")
    for item in tqdm(uncertain_items):
        prompt = PROMPT_TEMPLATE.format(item['question'])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            prob = torch.max(torch.softmax(outputs.logits[0, -1, :], dim=-1)).item()
            conf_scores[item['type']].append(prob)
            item['prompt'] = prompt

    # Save the mean baseline vector (x_base)
    if activations_baseline:
        x_base = torch.stack(activations_baseline).mean(dim=0)
        torch.save(x_base, "data/common_certainty_baseline.pt")
        print(f"Generated x_base from {len(activations_baseline)} high-confidence samples.")

    # Save the specific inputs used to create the baseline
    with open("data/baseline_inputs.json", "w", encoding="utf-8") as f:
        json.dump(verified_baseline_inputs, f, indent=4)

    # Save the mixed uncertainty study dataset (JSONL)
    with open("data/uncertainty_study_dataset.jsonl", "w") as f:
        for item in uncertain_items:
            f.write(json.dumps(item) + "\n")

    # Final Report
    avg_conf = {k: (sum(v) / len(v) if v else 0) for k, v in conf_scores.items()}
    print("\n" + "=" * 40)
    print("CALIBRATION REPORT (VERIFIED)")
    print("=" * 40)
    print(f"Baseline (n={len(verified_baseline_inputs)}):  {avg_conf['baseline']:.4f}")
    print(f"Epistemic (n={len(conf_scores['epistemic'])}): {avg_conf['epistemic']:.4f}")
    print(f"Aleatoric (n={len(conf_scores['aleatoric'])}): {avg_conf['aleatoric']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    process_and_compare()