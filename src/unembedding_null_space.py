import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# 1. Load the model
# Using GPT-2 Small as per the paper's primary case study
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# 2. Extract and Center the Unembedding Matrix (W_U)
# W_U shape: [d_model, d_vocab] -> [768, 50257]
W_U = model.W_U

# Centering is crucial: softmax is translation invariant,
# so the mean of W_U across the vocab dimension doesn't affect predictions.
# By removing the mean, we isolate the meaningful directions.
W_U_centered = W_U - W_U.mean(dim=1, keepdim=True)

# 3. Perform SVD on the Transpose
# We analyze W_U.T [50257, 768] to find directions in the 768-dim residual stream
# that have the least impact on the 50,257-dim output space.
U, S, Vh = torch.linalg.svd(W_U_centered.T, full_matrices=False)

# 4. Visualize the Singular Values
# We plot the tail of the spectrum (bottom 50 values) to find the "elbow"
plt.figure(figsize=(10, 6))
tail_size = 50
indices = range(768 - tail_size, 768)
values = S.cpu().detach().numpy()[-tail_size:]

plt.plot(indices, values, marker='o', linestyle='-', color='b', label='Singular Values')

# Draw a boundary at k=12 (index 756) to verify the paper's claim
plt.axvline(x=768 - 12, color='r', linestyle='--', label='k=12 (Stolfo et al.)')

plt.title("Bottom Singular Values of GPT-2 Small Unembedding")
plt.xlabel("Singular Value Index")
plt.ylabel("Magnitude")
plt.yscale('log') # Log scale helps see the magnitude of the drop
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

# 5. Print the numerical values for the suspected null space
print("Last 15 Singular Values (The 'Effective Null Space'):")
for i, val in enumerate(S[-15:]):
    idx = 768 - 15 + i
    print(f"Index {idx}: {val.item():.6f}")

# 6. Construct the Projection Matrix P_perp
# If the drop-off confirms k=12, we take the last 12 rows of Vh
k = 12
V_null_basis = Vh[-k:] # Shape: [12, 768]
P_perp = V_null_basis.T @ V_null_basis # Shape: [768, 768]

# Save the matrix to use in your activation caching script
torch.save(P_perp, "P_perp_gpt2_small.pt")
print("\nProjection matrix P_perp saved to P_perp_gpt2_small.pt")