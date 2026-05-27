import pickle
import numpy as np
import matplotlib.pyplot as plt

file_path = "/home/.../results_RL_csi_adaptive/reward_history_training.pkl"

with open(file_path, "rb") as f:
    reward_history = pickle.load(f)

# -------- For Each user seperately ------
# reward_history = np.array(reward_history)
# def moving_average(data, window=200):
#     return np.convolve(data, np.ones(window)/window, mode='valid')

# plt.figure(figsize=(10, 5))

# for k in range(reward_history.shape[1]):
#     smoothed = moving_average(reward_history[:, k])
#     plt.plot(smoothed, label=f"User {k}")

# plt.xlabel("Step")
# plt.ylabel("Reward")
# plt.title("Reward per User During Training")
# plt.legend()
# plt.grid()
# plt.savefig("Training.png", dpi=300)
# plt.show()

# ------- Averaging over users -------
reward_history = np.array(reward_history)
avg_reward = reward_history.mean(axis=1)
def moving_average(data, window=200):
    return np.convolve(data, np.ones(window)/window, mode='valid')

smoothed = moving_average(avg_reward, window=5000)

plt.figure(figsize=(10, 5))
plt.plot(smoothed)

plt.xlabel("Step")
plt.ylabel("Smoothed Avg Reward")
plt.title("Smoothed Average Reward Across Users")
plt.grid()
plt.savefig("Training.png", dpi=300)
plt.show()
