import torch
import json
import os
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# 2. Load Geometric Anchor
x_base = torch.load("data/common_certainty_baseline.pt").to(device)

# --- Compute P_perp (Projection to Null Space) ---
W_U = model.transformer.wte.weight.detach().cpu()
W_U_centered = W_U - W_U.mean(dim=0)
U, S, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
k = 12
V_null = Vh[-k:, :].to(device)
P_perp = (V_null.t() @ V_null)


def run_analysis():
    dataset = []
    with open("data/uncertainty_study_dataset.jsonl", "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    results = {
        "epistemic_ratios": [],
        "aleatoric_ratios": [],
        "epistemic_vectors": [],
        "aleatoric_vectors": []
    }

    print("Analyzing geometry and null-space saturation...")
    for item in tqdm(dataset):
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            x_i = outputs.hidden_states[-1][0, -1, :]  # Final activation

            delta_x = x_i - x_base
            delta_V_hat = delta_x @ P_perp

            # 1. Norm Ratio Calculation (How much is written to the Null Space?)
            norm_original = torch.norm(delta_x).item()
            norm_projected = torch.norm(delta_V_hat).item()
            ratio = (norm_projected / norm_original) * 100

            if item['label'] == 0:
                results["epistemic_ratios"].append(ratio)
                results["epistemic_vectors"].append(delta_V_hat.cpu().numpy())
            else:
                results["aleatoric_ratios"].append(ratio)
                results["aleatoric_vectors"].append(delta_V_hat.cpu().numpy())

    # --- RESULTS REPORTING ---
    avg_epi_ratio = np.mean(results["epistemic_ratios"])
    avg_ale_ratio = np.mean(results["aleatoric_ratios"])

    # Prepare for cosine similarity
    epi_vecs = np.stack(results["epistemic_vectors"])
    ale_vecs = np.stack(results["aleatoric_vectors"])

    # Intra-class similarity (how consistent is each class?)
    sim_epi_epi = cosine_similarity(epi_vecs).mean()
    sim_ale_ale = cosine_similarity(ale_vecs).mean()

    # Inter-class similarity (how different are the classes from each other?)
    sim_epi_ale = cosine_similarity(epi_vecs, ale_vecs).mean()

    print("\n" + "=" * 40)
    print("GEOMETRIC ANALYSIS REPORT")
    print("=" * 40)
    print(f"NULL SPACE SATURATION (Norm %):")
    print(f" - Epistemic: {avg_epi_ratio:.2f}%")
    print(f" - Aleatoric: {avg_ale_ratio:.2f}%")
    print("-" * 40)
    print(f"COSINE SIMILARITY (Directional Alignment):")
    print(f" - Epistemic <-> Epistemic: {sim_epi_epi:.4f}")
    print(f" - Aleatoric <-> Aleatoric: {sim_ale_ale:.4f}")
    print(f" - Epistemic <-> Aleatoric: {sim_epi_ale:.4f}")
    print("=" * 40)

    # INTERPRETATION
    if sim_epi_epi > sim_epi_ale and sim_ale_ale > sim_epi_ale:
        print("CONCLUSION: Success! The classes are more similar to themselves than to each other.")
        print("This confirms distinct geometric directions for each uncertainty type.")
    else:
        print("CONCLUSION: Low separation detected. The directions might be overlapping.")
    print("=" * 40)


if __name__ == "__main__":
    run_analysis()