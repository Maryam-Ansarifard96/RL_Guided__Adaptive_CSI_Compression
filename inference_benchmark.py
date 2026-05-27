import torch.nn.functional as F
import os
import time
import math
import random
from collections import namedtuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from scipy.io import loadmat
from scipy.io import savemat
from ATH_baseline import ATHController

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
def build_joint_state_for_user(state_list, user_idx):
    """
    Build a user-specific full joint state by placing the current user's local
    state first, followed by the other users' states in a fixed order.
    """
    ordered = [state_list[user_idx]]
    ordered.extend(state_list[j] for j in range(len(state_list)) if j != user_idx)
    return np.concatenate(ordered, axis=0).astype(np.float32)

def benchmark_dqn_inference(q_net, state_dim, device, num_warmup=500, num_runs=20000): # "Policy inference latency”
    q_net.eval()

    dummy_state = torch.randn(1, state_dim, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = q_net(dummy_state)

    # Timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = q_net.act(dummy_state)
    end = time.perf_counter()

    avg_latency_us = (end - start) / num_runs * 1e6
    return avg_latency_us

def benchmark_rl_full(env, agent, num_runs=1000):
    state = env.reset()

    start = time.perf_counter()

    for _ in range(num_runs):
        joint_states = np.stack([
            build_joint_state_for_user(state, k) for k in range(env.K)
        ])
        state_tensor = torch.tensor(joint_states, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            q_values = agent.q_net(state_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        state, _, done, _ = env.step(actions)  

        if done:
            state = env.reset()

    end = time.perf_counter()

    return (end - start) / num_runs * 1e6

def benchmark_ath_batch_inference(env, num_warmup=500, num_runs=20000):
    
    K = env.K
    controllers = [ATHController() for _ in range(K)]
    for c in controllers:
        c.reset()

    # Dummy inputs for batch
    dummy_snr = np.random.uniform(0, 30, size=K)
    dummy_nmse = np.random.uniform(0, 0.5, size=K)

    # --- Warm-up ---
    for _ in range(num_warmup):
        actions = np.zeros(K, dtype=int)
        for k in range(K):
            actions[k] = controllers[k].select_action(dummy_snr[k], dummy_nmse[k])

    # --- Timing ---
    start = time.perf_counter()
    for _ in range(num_runs):
        actions = np.zeros(K, dtype=int)
        for k in range(K):
            actions[k] = controllers[k].select_action(dummy_snr[k], dummy_nmse[k])
    end = time.perf_counter()

    avg_latency_us = (end - start) / num_runs * 1e6
    return avg_latency_us

def benchmark_dnn_inference(models, compression_list, state_dim, mu, sigma, device, num_warmup=200, num_runs=5000):
    
    # Average per-user stats for a representative normalization
    mu_ref = torch.stack(mu).mean(dim=0)     # (feature_dim,)
    sigma_ref = torch.stack(sigma).mean(dim=0)  # (feature_dim,)
    any_model = next(iter(models.values()))
    csi_dim = np.prod(any_model.input_shape)

    dummy_x = torch.randn(1, csi_dim, device=device)
    dummy_x_norm = (dummy_x - mu_ref) / sigma_ref

    sorted_crs = sorted(compression_list)
    NMSE_THRESHOLD = 0.1

    # Set all models to eval
    for model in models.values():
        model.eval()

    with torch.no_grad():
        for _ in range(num_warmup):
            for cr in sorted_crs:
                model = models[cr]
                recon_n, _ = model(dummy_x_norm)
                _ = recon_n.sum()  # prevent lazy eval

    # --- Timing ---
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            for cr in sorted_crs:
                model = models[cr]
                recon_n, _ = model(dummy_x_norm)

                # Fake NMSE check (we don't care about correctness here)
                if torch.rand(1).item() < 0.5:
                    break
    end = time.perf_counter()

    avg_latency_us = (end - start) / num_runs * 1e6
    return avg_latency_us
def benchmark_fixed_ratio_inference(env, fixed_ratio=1/8, num_runs=100):
    
    K = env.K
    if fixed_ratio not in COMPRESSION_RATIOS:
        raise ValueError("Fixed ratio must be one of COMPRESSION_RATIOS")
    action_idx = COMPRESSION_RATIOS.index(fixed_ratio) + 1
    state = env.reset()
    for _ in range(10):
        actions = np.ones(K, dtype=int) * action_idx
        state, _, done, _ = env.step(actions)
        if done:
            state = env.reset()
    start = time.perf_counter()
    for _ in range(num_runs):
        actions = np.ones(K, dtype=int) * action_idx
        state, _, done, _ = env.step(actions)

        if done:
            state = env.reset()

    end = time.perf_counter()
    avg_latency_us = (end - start) / num_runs * 1e6
    return avg_latency_us