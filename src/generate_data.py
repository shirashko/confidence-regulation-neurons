import pandas as pd
from datasets import load_dataset
import os

# 1. Load PopQA for Epistemic Uncertainty (Knowledge Gaps)
print("Loading PopQA...")
pop_qa = load_dataset("akariasai/PopQA", split="test")
pop_df = pd.DataFrame(pop_qa)

# Filter for long-tail entities where monthly pageviews < 100
epistemic_all = pop_df[pop_df['s_pop'] < 100].copy()
print(f"Found {len(epistemic_all)} epistemic candidates.")

# 2. Load AmbigQA for Aleatoric Uncertainty (Ambiguity)
print("Loading AmbigQA...")
ambig_qa = load_dataset("sewon/ambig_qa", "full", split="train")
ambig_df = pd.DataFrame(ambig_qa)

def is_ambiguous(ann):
    """Handles the nested dictionary/list structure of AmbigQA annotations."""
    try:
        # Access the 'type' key and check the first element
        return ann['type'][0] == 'multipleQAs'
    except (KeyError, IndexError, TypeError):
        return False

# Filter for questions identified as having multiple valid interpretations
aleatoric_all = ambig_df[ambig_df['annotations'].apply(is_ambiguous)].copy()
print(f"Total aleatoric candidates found: {len(aleatoric_all)}")

# 3. Balancing the Dataset (1:1 Ratio)
# We use the size of the epistemic set (1,399) as the target
sample_size = len(epistemic_all)
if len(aleatoric_all) > sample_size:
    aleatoric_subset = aleatoric_all.sample(n=sample_size, random_state=42).copy()
else:
    aleatoric_subset = aleatoric_all.copy()

# 4. Finalize formatting and labels
def format_prompt(question):
    """Standardizes input for GPT-2 processing."""
    return f"Question: {question} Answer:"

# Prepare Epistemic (Label 0)
epistemic_final = epistemic_all[['id', 'question']].copy()
epistemic_final['prompt'] = epistemic_final['question'].apply(format_prompt)
epistemic_final['uncertainty_type'] = 'epistemic'
epistemic_final['label'] = 0

# Prepare Aleatoric (Label 1)
aleatoric_final = aleatoric_subset[['id', 'question']].copy()
aleatoric_final['prompt'] = aleatoric_final['question'].apply(format_prompt)
aleatoric_final['uncertainty_type'] = 'aleatoric'
aleatoric_final['label'] = 1

# Combine and Shuffle to prevent ordering bias for the probe
final_dataset = pd.concat([epistemic_final, aleatoric_final])
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Save to disk
os.makedirs("data", exist_ok=True)
output_path = "data/uncertainty_study_dataset.jsonl"
final_dataset.to_json(output_path, orient="records", lines=True)

print(f"\nSuccess! Final dataset created with {len(final_dataset)} total samples.")
print(f"File saved to: {os.path.abspath(output_path)}")

# 6. Quick Preview
print("\nSample from dataset:")
print(final_dataset[['prompt', 'uncertainty_type', 'label']].head())