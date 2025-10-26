import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig

# -------------------------------
# 1. RoPE waveform computation
# -------------------------------
def rope_waveform(base: float, d_head: int = 128, L: int = 4096, device="cpu"):
    """Compute average RoPE similarity waveform for a given base."""
    # Calculate half the dimension, as RoPE uses 2x2 blocks.
    half_dim = d_head // 2
    # Create an index vector [0, 1, 2, ..., half_dim - 1].
    idx = torch.arange(half_dim, device=device, dtype=torch.float32)
    # Calculate the frequency vector (theta) for each pair of dimensions.
    theta = base ** (-2 * idx / d_head)
    # Create the position vector [0, 1, 2, ..., L - 1].
    pos = torch.arange(L, device=device, dtype=torch.float32)
    # Compute the cosine of the outer product (position * frequency), then average across the dimension (-1).
    wave = torch.mean(torch.cos(torch.outer(pos, theta)), dim=-1)
    # The resulting vector is the average similarity waveform over all positions.
    return wave  # [L]

# -------------------------------
# 2. Peak / trough detection
# -------------------------------
def find_peaks_and_troughs(wave):
    grad = torch.sign(wave[1:] - wave[:-1])
    changes = grad[1:] - grad[:-1]
    peaks = (changes < 0).nonzero(as_tuple=False).squeeze(1) + 1
    troughs = (changes > 0).nonzero(as_tuple=False).squeeze(1) + 1
    return peaks, troughs

# -------------------------------
# 3. Complementarity score
# -------------------------------
def complementarity_score(P_c, T_c, P_j, T_j):
    if len(P_j) == 0 or len(T_j) == 0 or len(P_c) == 0 or len(T_c) == 0:
        return float("inf")
    d1 = torch.cdist(P_j.float().unsqueeze(1), T_c.float().unsqueeze(1)).min(dim=1).values.mean()
    d2 = torch.cdist(T_j.float().unsqueeze(1), P_c.float().unsqueeze(1)).min(dim=1).values.mean()
    return (d1 + d2).item()

# -------------------------------
# 4. Algorithm 1: Base selection
# -------------------------------
def select_rope_bases(B_min, B_max, S, N, d_head, L, device="cpu"):
    B_s = torch.arange(B_min, B_max + S, S)
    print(f"Total candidates: {len(B_s)}")

    selected = [B_min]
    wave = rope_waveform(B_min, d_head, L, device)
    P_c, T_c = find_peaks_and_troughs(wave)

    for step in range(1, N):
        best_base, best_score = None, float("inf")
        for B_j in B_s:
            if any(abs(B_j - b) < 1e-9 for b in selected):
                continue
            wave_j = rope_waveform(float(B_j), d_head, L, device)
            P_j, T_j = find_peaks_and_troughs(wave_j)
            score = complementarity_score(P_c, T_c, P_j, T_j)
            if score < best_score:
                best_score, best_base = score, float(B_j)

        selected.append(best_base)
        wave_best = rope_waveform(best_base, d_head, L, device)
        P_best, T_best = find_peaks_and_troughs(wave_best)
        P_c = torch.unique(torch.cat([P_c, P_best]))
        T_c = torch.unique(torch.cat([T_c, T_best]))

        print(f"[{step}] Added base={best_base:.2e} (score={best_score:.3f})")

    return selected

# -------------------------------
# 5. Main
# -------------------------------
if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    cfg = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    d_head = cfg.hidden_size // cfg.num_attention_heads
    B_train = cfg.rope_theta  # ~5e5

    bases = select_rope_bases(
        B_min=B_train,
        B_max=1000000,
        S=5000,
        N=7,
        d_head=d_head,
        L=8192,
        device=device,
    )

    print("Selected bases:", bases)

    # -------------------------------
    # 6. Visualization + Save as JPG
    # -------------------------------
    plt.figure(figsize=(10, 6))
    for B in bases:
        wave = rope_waveform(B, d_head=d_head, L=4096, device="cpu")
        plt.plot(wave.cpu().numpy(), label=f"B={B:.1e}")

    plt.title("RoPE Waveforms for Selected Bases (Llama 3 Config)")
    plt.xlabel("Δ position")
    plt.ylabel("Avg cos similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure as JPG
    output_file = "rope_waveforms_llama3.jpg"
    plt.savefig(output_file, format="jpg", dpi=300)
    print(f"Saved waveform plot → {output_file}")

    plt.show()