import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.io import savemat
class OnlineNormalizer:
    def __init__(self, K, feature_dim, init_mu=None, init_sigma=None, momentum=0.01):
        """
        momentum: how fast to adapt to new data (0=never update, 1=no memory)
        Lower momentum = more stable but slower to adapt to drift.
        """
        self.K = K
        self.momentum = momentum
        self.feature_dim = feature_dim
        self.count = np.zeros(K)
        if init_mu is not None and init_sigma is not None:
            self.mu = np.stack([init_mu[k].cpu().numpy() for k in range(K)]).astype(np.float32)
            self.sigma = np.stack([init_sigma[k].cpu().numpy() for k in range(K)]).astype(np.float32)
        else:
            self.mu = np.zeros((K, feature_dim), dtype=np.float32)
            self.sigma = np.ones((K, feature_dim),  dtype=np.float32)
        self.init_mu = self.mu.copy()
        self.init_sigma = self.sigma.copy()
        self.var = np.maximum(self.sigma ** 2, 1e-4).astype(np.float32)
        self.init_var = self.var.copy()

    def update(self, k, x_flat):
        """EMA update for user k with new CSI sample."""
        x_flat = np.asarray(x_flat, dtype=np.float32)
        effective_momentum = self.momentum
        self.count[k] += 1

        prev_mu = self.mu[k].copy()
        new_mu = (1.0 - effective_momentum) * prev_mu + effective_momentum * x_flat

        # Update the second central moment with the same EMA so scale changes
        # smoothly and remains compatible with the pretrained normalization.
        centered = x_flat - new_mu
        new_var = (1.0 - effective_momentum) * self.var[k] + effective_momentum * (centered ** 2)

        self.mu[k] = new_mu
        self.var[k] = np.maximum(new_var, 1e-4)
        self.sigma[k] = np.sqrt(self.var[k] + 1e-8).astype(np.float32)

    def reset_counts(self):
        """Call this in env.reset() so momentum re-adapts at episode start."""
        self.count = np.zeros(self.K)

    def reset_to_initial_stats(self):
        """Restore the original dataset statistics and clear adaptation counters."""
        self.mu = self.init_mu.copy()
        self.sigma = self.init_sigma.copy()
        self.var = self.init_var.copy()
        self.count = np.zeros(self.K)

    def get_tensors(self, k, device):
        mu_k    = torch.from_numpy(self.mu[k]).float().to(device)     # (feature_dim,)
        sigma_k = torch.from_numpy(self.sigma[k]).float().to(device)  # (feature_dim,)
        return mu_k, sigma_k
