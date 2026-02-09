import torch
import numpy as np
import json
from transformers import GPT2LMHeadModel
from sklearn.linear_model import LogisticRegression

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 2. Re-compute P_perp
W_U = model.transformer.wte.weight.detach().cpu()
W_U_centered = W_U - W_U.mean(dim=0)
_, _, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
k = 12
V_null = Vh[-k:, :].to(device)
P_perp = (V_null.t() @ V_null)

# 3. Load data and get Probe direction
with open("data/steering_vectors.json", "r") as f:
    data = json.load(f)
X = np.array([item["vector"] for item in data])
y = np.array([item["label"] for item in data])
probe = LogisticRegression(max_iter=1000).fit(X, y)
w_dir = torch.tensor(probe.coef_[0], dtype=torch.float32).to(device)


def find_influential_neurons():
    # Neurons identified in Stolfo et al. (2024) as entropy/confidence neurons in Layer 11
    paper_neurons = [584, 2123, 2032, 1245, 2921, 812, 1507]

    W_out = model.transformer.h[-1].mlp.c_proj.weight.detach()
    W_out_null = W_out @ P_perp
    scores = (W_out_null @ w_dir).cpu().numpy()

    # Get general top neurons
    top_aleatoric = np.argsort(scores)[-10:][::-1]
    top_epistemic = np.argsort(scores)[:10]

    print("\n" + "=" * 40)
    print("TOP REGULATION NEURONS (Layer 11)")
    print("=" * 40)
    print("Predicting ALEATORIC (Ambiguity):")
    for i, idx in enumerate(top_aleatoric):
        print(f" {i + 1}. Neuron #{idx} (Score: {scores[idx]:.4f})")

    print("-" * 40)
    print("Predicting EPISTEMIC (Knowledge Gap):")
    for i, idx in enumerate(top_epistemic):
        print(f" {i + 1}. Neuron #{idx} (Score: {scores[idx]:.4f})")

    print("\n" + "=" * 40)
    print("PAPER NEURONS CHECK")
    print("=" * 40)
    print(f"{'Neuron ID':<10} | {'Score':<10} | {'Likely Type'}")
    print("-" * 35)
    for idx in paper_neurons:
        score = scores[idx]
        ntype = "Aleatoric" if score > 0 else "Epistemic"
        # Star the ones that are very significant (> 2.0 in absolute value)
        star = "*" if abs(score) > 2.0 else ""
        print(f"{idx:<10} | {score:<10.4f} | {ntype} {star}")
    print("=" * 40)
    print("(* indicates high influence score > 2.0)")


if __name__ == "__main__":
    find_influential_neurons()