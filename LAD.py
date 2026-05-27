import torch.nn.functional as F
import os
import time
import math
import random
from collections import namedtuple
from collections import deque
import numpy as np
import h5py
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
from copy import deepcopy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Helper
def compute_nmse(recon, target):
    # both complex numpy arrays
    num = np.sum(np.abs(target - recon)**2)
    den = np.sum(np.abs(target)**2) + 1e-12
    return num/den

# ------------------------------------------------------------------------------
# PAPER IMPLEMENTATION: Adaptive DNN-based CSI Feedback with Quantization
# ------------------------------------------------------------------------------
class QuantizationLayer(nn.Module):
    """
    Mu-law Non-uniform Quantization (Eq. 6 in the paper).
    """
    def __init__(self, bits=4, mu=255.0):
        super(QuantizationLayer, self).__init__()
        self.bits = bits
        self.mu = float(mu)
        self.L = 2 ** bits
        
    def forward(self, x):
        # Forward pass: Quantize
        # 1. Normalization (Paper uses tanh or similar to bound input)
        # We assume input is roughly in [-1, 1] after Tanh in encoder
        
        # 2. Companding: f(x) = sgn(x) * ln(1 + mu|x|) / ln(1 + mu)
        num = torch.log(1 + self.mu * torch.abs(x))
        denom = math.log(1 + self.mu)
        companded = torch.sign(x) * (num / denom)
        
        # 3. Uniform Quantization
        # Map [-1, 1] -> [0, L-1]
        x_q = torch.round((companded + 1) / 2 * (self.L - 1))
        
        # 4. De-quantization (for gradient/reconstruction)
        x_dq = (x_q / (self.L - 1)) * 2 - 1
        
        # 5. Expand (Inverse Companding)
        expanded = (torch.sign(x_dq) / self.mu) * ((1 + self.mu)**torch.abs(x_dq) - 1)
        
        # Straight-Through Estimator (STE) for backprop
        # During backward, gradients flow as if this layer was identity
        return (expanded - x).detach() + x
# AE of LAD baseline
class PaperAE(nn.Module):
    """
    Adapted for 2D input (Batch, 2, NumBSant, Nsub).
    """
    def __init__(self, num_ant, n_sub, cr, quantization_bits=None):
        super(PaperAE, self).__init__()
        self.input_shape = (2, num_ant, n_sub)
        self.total_dim = 2 * num_ant * n_sub
        self.M = int(self.total_dim * cr) # Codeword length
        self.quantize = (quantization_bits is not None)
        
        # --- Encoder ---
        # The paper uses Conv3D, but for (Ant, Sub) data, Conv2D is equivalent/better
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # Downsample
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        
        # Calculate Flatten Size
        # Input (H, W) -> Stride 2 -> (H/2, W/2)
        h_out = math.ceil(num_ant / 2)
        w_out = math.ceil(n_sub / 2)
        self.flat_features = 16 * h_out * w_out
        
        # Paper Innovation: Replace final conv with Dense for arbitrary CR
        self.encoder_fc = nn.Linear(self.flat_features, self.M)
        
        if self.quantize:
            self.quantizer = QuantizationLayer(bits=quantization_bits)
            
        # --- Decoder ---
        self.decoder_fc = nn.Linear(self.M, self.total_dim) # Direct mapping per paper simple decoder or RefineNet
        
        # The paper mentions a "RefineNet" or "Dequantization Module" (Residual)
        # We implement a RefineNet structure
        self.refinenet = nn.Sequential(
            nn.Linear(self.total_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.total_dim)
        )

    def forward(self, x):
        # x input: (Batch, Total_Dim) flat or (Batch, 2, H, W)
        # Reshape to image for Conv
        batch_size = x.shape[0]
        x_img = x.view(batch_size, 2, self.input_shape[1], self.input_shape[2])
        
        # Encode
        feat = self.encoder_conv(x_img)
        feat_flat = feat.view(batch_size, -1)
        codeword = self.encoder_fc(feat_flat)
        
        # Quantize
        if self.quantize:
            codeword = self.quantizer(codeword)
            
        # Decode
        recon_rough = self.decoder_fc(codeword)
        
        # Refine
        residual = self.refinenet(recon_rough)
        recon = recon_rough + residual
        
        return recon, codeword

class CRClassifier(nn.Module):
    """
    Predicts which compression ratio should be used.
    """
    def __init__(self, input_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
def build_cr_classifier_dataset(data, models, mu, sigma, compression_list, margin=0.005):
    """
    Build training data for the CR classifier.

    Label = best CR for each sample based on reconstruction NMSE.
    Keep only samples where the best CR is clearly better than second-best,
    similar to the paper's thresholded classifier-label generation.
    """
    T, K = data['CSI'].shape[0], data['CSI'].shape[1]

    def flatten_real_imag(H):
        return np.concatenate(
            [np.real(H.reshape(-1)), np.imag(H.reshape(-1))]
        ).astype(np.float32)

    X_list, y_list = [], []

    for t in range(T):
        for k in range(K):
            H = data['CSI'][t, k, ...]
            x_flat = flatten_real_imag(H)

            mu_k = mu[k]
            sigma_k = sigma[k]
            x_t = torch.from_numpy(x_flat).float().unsqueeze(0).to(DEVICE)
            x_norm = (x_t - mu_k) / sigma_k

            nmse_per_cr = []
            with torch.no_grad():
                for cr in compression_list:
                    model = models[cr]
                    model.eval()
                    recon_n, _ = model(x_norm)

                    recon_n_np = recon_n.cpu().numpy().flatten()
                    x_norm_np = x_norm.cpu().numpy().flatten()
                    N = len(recon_n_np) // 2
                    recon_c = recon_n_np[:N] + 1j * recon_n_np[N:]
                    target_c = x_norm_np[:N] + 1j * x_norm_np[N:]
                    nmse = compute_nmse(recon_c, target_c)
                    nmse_per_cr.append(nmse)

            order = np.argsort(nmse_per_cr)
            best_idx = int(order[0])
            second_idx = int(order[1])

            if (nmse_per_cr[second_idx] - nmse_per_cr[best_idx]) >= margin:
                X_list.append(x_flat)
                y_list.append(best_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y
def train_cr_classifier(data, models, mu, sigma, compression_list, epochs=20, batch_size=256, margin=0.005):
    X, y = build_cr_classifier_dataset(
        data=data,
        models=models,
        mu=mu,
        sigma=sigma,
        compression_list=compression_list,
        margin=margin
    )

    if len(X) == 0:
        raise RuntimeError("No classifier samples survived. Reduce margin.")

    input_dim = X.shape[1]
    classifier = CRClassifier(input_dim, len(compression_list)).to(DEVICE)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = classifier(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_seen += xb.size(0)

        print(f"[CR Classifier] Epoch {ep+1}/{epochs} | "
              f"Loss={total_loss/total_seen:.6f} | Acc={total_correct/total_seen:.4f}")

    return classifier
def train_paper_baseline_models(data, mu, sigma, ratios, epochs=50):
    """
    Trains a set of PaperAE models, one for each Compression Ratio.
    Returns a dictionary of trained models.
    """
    print(f"\n--- Training Paper Baseline Models (Adaptive Dictionary) ---")
    models = {}
    
    T, K = data['CSI'].shape[0], data['CSI'].shape[1]
    def flatten_real_imag(H):
        return np.concatenate(
            [np.real(H.reshape(-1)), np.imag(H.reshape(-1))]
        ).astype(np.float32)
    # Normalize each user's samples with their own stats, then pool
    all_chunks = []
    for k in range(K):
        user_H_flat = np.stack([
            flatten_real_imag(data['CSI'][t, k, ...])
            for t in range(T)
        ])  # (T, feature_dim)
        user_H_t = torch.from_numpy(user_H_flat).float().to(DEVICE)
        user_H_t_norm = (user_H_t - mu[k]) / sigma[k]
        all_chunks.append(user_H_t_norm)

    all_H_t_norm = torch.cat(all_chunks, dim=0)  # (T*K, feature_dim)
    
    dataset = torch.utils.data.TensorDataset(all_H_t_norm)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    NumBSant = data['CSI'].shape[2]
    Nsub = data['CSI'].shape[4]
    
    for cr in ratios:
        print(f"Training Paper Model for CR={cr}...")
        # Paper uses quantization (e.g., 5 bits). We add it here.
        model = PaperAE(NumBSant, Nsub, cr, quantization_bits=5).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(DEVICE)
                optimizer.zero_grad()
                recon, _ = model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch+1) % 25 == 0:
                print(f"  Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.5f}")
                
        models[cr] = model 
    return models


# def evaluate_paper_adaptive_baseline(data, models, mu, sigma, compression_list):
    
#     print("\n--- Evaluating Paper Adaptive Baseline ---")
#     K = data['CSI'].shape[1]
#     T = data['CSI'].shape[0]
    
#     env = MultiUserCSIEnv(data, mu_sigma=(mu, sigma), base_checkpoint=pretrain_paths, allow_finetune=False)
#     env.normalizer.reset_counts()
#     rewards_hist = []
#     actions_hist = [] 
#     logger = RewardLogger(name="paper_baseline")
#     last_ratio = [compression_list[0]] * K
#     # NMSE Threshold for "Acceptable Quality" (Paper logic)
#     # If NMSE < THRESHOLD with low CR, pick low CR. Else pick high CR.
#     NMSE_THRESHOLD = 0.1 
    
#     with torch.no_grad():
#         for t in range(T):
#             if t % 100 == 0: print(f"Evaluating Slot {t}/{T}")
            
#             H_slot = env._get_raw_csi_for_all_users(t) # K x Ant x 1 x Sub
#             H_true_slot = env._get_csi_slot(t) 
#             H_est_paper = np.zeros_like(H_true_slot, dtype=complex)
            
#             slot_rewards = []
#             slot_actions = []
#             final_nmse_per_user = {}
#             final_cr_per_user   = {}
#             for k in range(K):
#                 # Prepare Input
#                 rawH = H_slot[k, ...]
#                 x_flat = env._flatten_real_imag(rawH)
#                 # --- Online normalization ---
#                 mu_k, sigma_k = env.normalizer.get_tensors(k, DEVICE)
#                 x_t = torch.from_numpy(x_flat).float().unsqueeze(0).to(DEVICE)
#                 x_norm = (x_t - mu_k) / sigma_k
#                 env.normalizer.update(k, x_flat)
#                 # --- Adaptive Selection Logic ---
#                 # We try CRs from lowest (most compression) to highest.
#                 # We pick the first one that satisfies NMSE < Threshold.
#                 # If none, pick the highest.
                
#                 selected_model = None
#                 selected_cr = compression_list[0]
#                 selected_action_idx = 1 # 1-based index matching RL
                
#                 # Check models in order of increasing fidelity (decreasing compression ratio)
#                 # List is [1/4, 1/8...] -> 1/4 is High Fidelity, 1/32 is Low.
#                 # We want to use MINIMUM bits (Lowest CR like 1/32) if possible.
#                 # So iterate from smallest CR (last in list) to largest CR (first in list).
                
#                 sorted_crs = sorted(compression_list) # e.g. [1/32, 1/16, 1/8, 1/4]
                
#                 best_recon_norm = None
                
#                 # Paper "Classifier" mimic:
#                 # Based on SNR? Or try-all? 
#                 # "The classifier determines the CR... to adaptively output suitable CR"
#                 # We'll use a strong baseline: Try them and pick best tradeoff.
                
#                 final_cr = sorted_crs[0]
#                 final_nmse = 1.0
                
#                 # Strategy: Pick lowest CR where NMSE < 0.15 (typical threshold)
#                 # If SNR is very low, maybe we don't care about high accuracy?
                
#                 found = False
#                 mu_np    = mu_k.cpu().numpy()
#                 sigma_np = sigma_k.cpu().numpy()
#                 snr_k = float(data['userSNR_dB'][t, k])
#                 for cr in sorted_crs: 
#                     model = models[cr]
#                     model.eval()
#                     recon_n, _ = model(x_norm)
                    
#                     # NMSE in normalized space (scale-invariant)
#                     recon_n_np   = recon_n.cpu().numpy().flatten()
#                     x_norm_np    = x_norm.cpu().numpy().flatten()
#                     N            = len(recon_n_np) // 2
#                     recon_c_norm = recon_n_np[:N] + 1j * recon_n_np[N:]
#                     target_c_norm = x_norm_np[:N] + 1j * x_norm_np[N:]
#                     nmse = compute_nmse(recon_c_norm, target_c_norm)

#                     if nmse < NMSE_THRESHOLD:
#                         final_cr = cr
#                         final_nmse = nmse
#                         recon_raw = recon_n_np * sigma_np + mu_np
#                         H_est_paper[k, :, :] = (
#                             recon_raw[:N] + 1j * recon_raw[N:]
#                         ).reshape(H_true_slot.shape[1], H_true_slot.shape[2])
#                         best_recon_norm = recon_n
#                         found = True
#                         break # Found sufficient compression
                
#                 if not found:
#                     # Use highest fidelity
#                     final_cr = sorted_crs[-1]
#                     model = models[final_cr]
#                     recon_n, _ = model(x_norm)
#                     recon_n_np  = recon_n.cpu().numpy().flatten()
#                     x_norm_np   = x_norm.cpu().numpy().flatten()
#                     N           = len(recon_n_np) // 2
#                     recon_c_norm  = recon_n_np[:N] + 1j * recon_n_np[N:]
#                     target_c_norm = x_norm_np[:N] + 1j * x_norm_np[N:]
#                     final_nmse    = compute_nmse(recon_c_norm, target_c_norm)
#                     recon_raw     = recon_n_np * sigma_np + mu_np
#                     H_est_paper[k, :, :] = (
#                             recon_raw[:N] + 1j * recon_raw[N:]
#                         ).reshape(H_true_slot.shape[1], H_true_slot.shape[2])
                    
#                 final_nmse_per_user[k] = final_nmse
#                 final_cr_per_user[k]   = final_cr

#             sinr_users, rate = compute_sinr_from_csi(H_true_slot, H_est_paper, precoder_type="zf")
#             slot_payload_bits = sum(env._payload_bits(final_cr_per_user[k]) for k in range(K))
#             overflow = max(0.0, slot_payload_bits - env.max_total_bits)
#             budget_penalty_per_user = BUDGET_PENALTY_SCALE * overflow / (env.max_total_bits + 1e-12)

#             for k in range(K):
#                 rate_k   = np.log2(1.0 + sinr_users[k])
#                 cr_k     = final_cr_per_user[k]
#                 nmse_log = np.log1p(final_nmse_per_user[k])
#                 bit_cost_k = env._bit_cost_norm(cr_k)
#                 switch_cost = SWITCH_PENALTY if cr_k != last_ratio[k] else 0.0
#                 reward   = (
#                     GAMMA_THROUGHPUT * rate_k
#                     - ALPHA_NMSE    * nmse_log
#                     - BETA_BITS     * bit_cost_k
#                     - switch_cost
#                     - budget_penalty_per_user
#                 )
#                 slot_rewards.append(reward)
#                 reward_dict = compute_reward_terms(
#                     rate=rate_k,
#                     nmse=final_nmse_per_user[k],
#                     bits=bit_cost_k,
#                     did_finetune=False,
#                     budget_penalty=budget_penalty_per_user,
#                     switch_cost=switch_cost
#                 )
#                 logger.log({k: reward_dict})
#                 try:
#                     act_idx = compression_list.index(cr_k) + 1
#                 except ValueError:
#                     act_idx = 1
#                 slot_actions.append(act_idx)
#                 last_ratio[k] = cr_k
                
#             rewards_hist.append(slot_rewards)
#             actions_hist.append(slot_actions)
#     logger.save(SAVE_DIR)        
#     return np.array(rewards_hist), np.array(actions_hist)