import torch
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# 2. Load Geometric Anchor
x_base = torch.load("data/common_certainty_baseline.pt").to(device)

# --- Compute P_perp (Projection to Null Space) ---
# SVD is done on CPU due to MPS limitations, then moved back to device
W_U = model.transformer.wte.weight.detach().cpu()
W_U_centered = W_U - W_U.mean(dim=0)

print("Computing SVD (this might take a moment on CPU)...")
U, S, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)

# Bottom k=12 dimensions represent the effective null space
k = 12
V_null = Vh[-k:, :]
P_perp = (V_null.t() @ V_null).to(device)

def run_experiment():
    dataset = []
    # Ensure the file exists
    if not os.path.exists("data/uncertainty_study_dataset.jsonl"):
        print("Error: data/uncertainty_study_dataset.jsonl not found!")
        return

    with open("data/uncertainty_study_dataset.jsonl", "r") as f:
        for line in f:
            dataset.append(json.loads(line))

    final_vectors = []
    print(f"Projecting {len(dataset)} samples into the Null Space...")

    for item in tqdm(dataset):
        inputs = tokenizer(item['prompt'], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # x_i: Final layer activation at the last token position
            x_i = outputs.hidden_states[-1][0, -1, :]

            # Differential Calculation: Subtracting the verified certainty baseline
            delta_x = x_i - x_base
            # Null Space Projection: Isolating output-agnostic regulation signals
            delta_V_hat = delta_x @ P_perp

            final_vectors.append({
                "vector": delta_V_hat.cpu().numpy().tolist(),
                "label": item["label"],  # 0: Epistemic, 1: Aleatoric
                "type": item.get("type", "unknown") # Safe access to 'type'
            })

    with open("data/steering_vectors.json", "w") as f:
        json.dump(final_vectors, f)
    print("Success: Steering vectors saved to data/steering_vectors.json")


if __name__ == "__main__":
    run_experiment()