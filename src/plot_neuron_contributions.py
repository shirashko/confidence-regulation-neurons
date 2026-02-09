import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2LMHeadModel
from sklearn.linear_model import LogisticRegression

# 1. Setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# 2. Re-compute Projection Matrix P_perp
W_U = model.transformer.wte.weight.detach().cpu()
W_U_centered = W_U - W_U.mean(dim=0)
_, _, Vh = torch.linalg.svd(W_U_centered, full_matrices=False)
k = 12
V_null = Vh[-k:, :].to(device)
P_perp = (V_null.t() @ V_null)

# 3. Load data and re-fit the probe to get the direction (w_dir)
with open("data/steering_vectors.json", "r") as f:
    data = json.load(f)
X = np.array([item["vector"] for item in data])
y = np.array([item["label"] for item in data])
probe = LogisticRegression(max_iter=1000).fit(X, y)
w_dir = torch.tensor(probe.coef_[0], dtype=torch.float32).to(device)

# 4. Calculate scores for ALL 3072 neurons
W_out = model.transformer.h[-1].mlp.c_proj.weight.detach()
W_out_null = W_out @ P_perp
scores = (W_out_null @ w_dir).cpu().numpy()

# 5. Plotting the Histogram
plt.figure(figsize=(10, 6))
sns.histplot(scores, bins=100, kde=True, color='skyblue', edgecolor='black')

# Add lines for the top neurons to highlight them
plt.axvline(x=np.max(scores), color='red', linestyle='--', label=f'Top Aleatoric (Max: {np.max(scores):.2f})')
plt.axvline(x=np.min(scores), color='blue', linestyle='--', label=f'Top Epistemic (Min: {np.min(scores):.2f})')

plt.title("Distribution of Neuron Contribution Scores (Layer 11)", fontsize=14)
plt.xlabel("Contribution Score (Alignment with Uncertainty Direction)", fontsize=12)
plt.ylabel("Number of Neurons", fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Save the plot
plt.tight_layout()
plt.savefig("results/neuron_scores_distribution.png")
plt.show()

print("Distribution plot saved to results/neuron_scores_distribution.png")