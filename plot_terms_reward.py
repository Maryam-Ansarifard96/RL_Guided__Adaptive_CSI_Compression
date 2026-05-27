import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless / no GUI
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat_path = "/home/maryam/results_RL_csi_adaptive/rl_train.mat"
save_dir = "reward_term_figs_train"
os.makedirs(save_dir, exist_ok=True)

EPISODES = 10
STEPS = 5599
USERS = 2

def to_1d(x):
    return np.asarray(x).squeeze()

def moving_avg(x, w=100):
    x = np.asarray(x)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")

def save_line_plot(y, title, xlabel, ylabel, out_path, smooth_window=200):
    plt.figure(figsize=(12, 4))
    plt.plot(y, alpha=0.3, label="raw")

    if smooth_window is not None and len(y) >= smooth_window:
        ys = moving_avg(y, smooth_window)
        plt.plot(np.arange(len(ys)), ys, label=f"moving avg ({smooth_window})")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_episode_overlay(arr_2d, title, ylabel, out_path):
    # arr_2d: [episodes, steps]
    plt.figure(figsize=(12, 5))
    for ep in range(arr_2d.shape[0]):
        plt.plot(arr_2d[ep], alpha=0.75, label=f"ep {ep+1}")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_episode_mean_std(arr_2d, title, ylabel, out_path):
    # arr_2d: [episodes, steps]
    mean_curve = arr_2d.mean(axis=0)
    std_curve = arr_2d.std(axis=0)
    x = np.arange(arr_2d.shape[1])

    plt.figure(figsize=(12, 5))
    plt.plot(x, mean_curve, label="mean across episodes")
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.25, label="±1 std")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================
# LOAD AND RESHAPE
# ==========================
data = loadmat(mat_path)

nmse_penalty_flat = to_1d(data["nmse"])
expected_len = EPISODES * STEPS * USERS

if nmse_penalty_flat.size != expected_len:
    raise ValueError(
        f"nmse length = {nmse_penalty_flat.size}, expected = {expected_len}"
    )

nmse_penalty = nmse_penalty_flat.reshape(EPISODES, STEPS, USERS)

# Invert log1p(nmse) -> nmse
actual_nmse = np.expm1(nmse_penalty)

print("nmse_penalty shape:", nmse_penalty.shape)
print("actual_nmse shape:", actual_nmse.shape)
print("actual_nmse min/max:", actual_nmse.min(), actual_nmse.max())

# optional numerical safety
actual_nmse = np.maximum(actual_nmse, 0.0)

# ==========================
# 1) GLOBAL FLATTENED PLOTS
# ==========================
for user in range(USERS):
    y = actual_nmse[:, :, user].reshape(-1)
    save_line_plot(
        y=y,
        title=f"Actual NMSE | User {user} | all episodes flattened",
        xlabel="global step",
        ylabel="NMSE",
        out_path=os.path.join(save_dir, f"actual_nmse_user{user}_global.png"),
        smooth_window=200
    )

joint_mean = actual_nmse.mean(axis=2).reshape(-1)
joint_sum = actual_nmse.sum(axis=2).reshape(-1)

save_line_plot(
    y=joint_mean,
    title="Actual NMSE | joint users mean | all episodes flattened",
    xlabel="global step",
    ylabel="NMSE mean",
    out_path=os.path.join(save_dir, "actual_nmse_joint_mean_global.png"),
    smooth_window=200
)

save_line_plot(
    y=joint_sum,
    title="Actual NMSE | joint users sum | all episodes flattened",
    xlabel="global step",
    ylabel="NMSE sum",
    out_path=os.path.join(save_dir, "actual_nmse_joint_sum_global.png"),
    smooth_window=200
)

# ==========================
# 2) PER-EPISODE OVERLAYS
# ==========================
for user in range(USERS):
    save_episode_overlay(
        arr_2d=actual_nmse[:, :, user],
        title=f"Actual NMSE | User {user} | per-episode overlay",
        ylabel="NMSE",
        out_path=os.path.join(save_dir, f"actual_nmse_user{user}_episode_overlay.png")
    )

joint_mean_2d = actual_nmse.mean(axis=2)
joint_sum_2d = actual_nmse.sum(axis=2)

save_episode_overlay(
    arr_2d=joint_mean_2d,
    title="Actual NMSE | joint users mean | per-episode overlay",
    ylabel="NMSE mean",
    out_path=os.path.join(save_dir, "actual_nmse_joint_mean_episode_overlay.png")
)

save_episode_overlay(
    arr_2d=joint_sum_2d,
    title="Actual NMSE | joint users sum | per-episode overlay",
    ylabel="NMSE sum",
    out_path=os.path.join(save_dir, "actual_nmse_joint_sum_episode_overlay.png")
)

# ==========================
# 3) MEAN ± STD ACROSS EPISODES
# ==========================
for user in range(USERS):
    save_episode_mean_std(
        arr_2d=actual_nmse[:, :, user],
        title=f"Actual NMSE | User {user} | mean ± std across episodes",
        ylabel="NMSE",
        out_path=os.path.join(save_dir, f"actual_nmse_user{user}_mean_std.png")
    )

save_episode_mean_std(
    arr_2d=joint_mean_2d,
    title="Actual NMSE | joint users mean | mean ± std across episodes",
    ylabel="NMSE mean",
    out_path=os.path.join(save_dir, "actual_nmse_joint_mean_mean_std.png")
)

save_episode_mean_std(
    arr_2d=joint_sum_2d,
    title="Actual NMSE | joint users sum | mean ± std across episodes",
    ylabel="NMSE sum",
    out_path=os.path.join(save_dir, "actual_nmse_joint_sum_mean_std.png")
)

print(f"Saved all actual NMSE figures to: {save_dir}")