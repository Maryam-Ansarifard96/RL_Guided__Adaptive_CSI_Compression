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
