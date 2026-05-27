import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import numpy as np

data     = loadmat('/home/maryam/results_RL_csi_adaptive/comparison_results_test.mat')
rl_actions   = data["rl_action"]
ath_actions  = data["ath_action"]
paper_actions = data["paper_actions"]
T, K = rl_actions.shape[0], rl_actions.shape[1]
fixed4_actions = 0.25 * np.ones([T, K])
fixed8_actions = 0.125 * np.ones([T, K])
fixed16_actions = 0.0625 * np.ones([T, K])
fixed32_actions = 0.03125 * np.ones([T, K])

def plot_switching_times(rl_actions, ath_actions=None, paper_actions=None,
                         fixed4_actions=None, fixed8_actions=None,
                         fixed16_actions=None, fixed32_actions=None):
    """
    Plot when switching happens over time.
    A switch is counted when action[t] != action[t-1] for a user.
    Inputs are arrays of shape [T, K].
    """

    def compute_switch_matrix(actions):
        actions = np.array(actions)   # [T, K]
        if len(actions.shape) != 2:
            raise ValueError("actions must have shape [T, K]")
        switch_mat = np.zeros_like(actions, dtype=int)
        switch_mat[1:] = (actions[1:] != actions[:-1]).astype(int)
        return switch_mat

    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    plot_idx = 1
    methods = [
        ("RL", rl_actions),
        ("ATH", ath_actions),
        ("Paper", paper_actions),
        ("Fixed 1/4", fixed4_actions),
        ("Fixed 1/8", fixed8_actions),
        ("Fixed 1/16", fixed16_actions),
        ("Fixed 1/32", fixed32_actions),
    ]

    valid_methods = [(name, acts) for name, acts in methods if acts is not None]

    n_plots = len(valid_methods)
    n_rows = int(np.ceil(n_plots / 2))

    for name, acts in valid_methods:
        switch_mat = compute_switch_matrix(acts)

        plt.subplot(n_rows, 2, plot_idx)
        sns.heatmap(
            switch_mat.T,
            cmap="Reds",
            cbar=True,
            xticklabels=False,
            yticklabels=[f"User {k}" for k in range(switch_mat.shape[1])]
        )
        plt.title(f"{name}: Switching Times")
        plt.xlabel("Time Slot")
        plt.ylabel("User")
        plot_idx += 1

    plt.tight_layout()
    plt.savefig("Switching_Times_Comparison.png", dpi=300)
    plt.show()

plot_switching_times(rl_actions, ath_actions, paper_actions,
                         fixed4_actions, fixed8_actions,
                         fixed16_actions, fixed32_actions)