# Compared to Baseline_23_3_v3, Pretrain one AE per CR ratio and load the matching one when RL switches:
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
from ATH_baseline import ATHController
from Normalizer import OnlineNormalizer
from LAD import train_cr_classifier, train_paper_baseline_models
from inference_benchmark import benchmark_dqn_inference, benchmark_ath_batch_inference, \
                                benchmark_rl_full, benchmark_dnn_inference, benchmark_fixed_ratio_inference
# --------------------------- Configuration ---------------------------------
DATASET_PATH = os.environ.get(
    "CSI_DATASET_PATH",
    "/home/maryam/CSI_compression/Agentic_ai_addaptive/CSI_GlobalScenario_CRTarget_K2_v2.mat"
) 

SAVE_DIR = "results_RL_csi_adaptive"
DATASET_TAG = os.path.splitext(os.path.basename(DATASET_PATH))[0]
SAVE_DIR = f"results_RL_csi_adaptive_{DATASET_TAG}"
os.makedirs(SAVE_DIR, exist_ok=True)
AE_CHECKPOINT_DIR = f"ae_checkpoints_{DATASET_TAG}"
os.makedirs(AE_CHECKPOINT_DIR, exist_ok=True)
AE_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "ae_base_pretrained.pt") 

BATCH_SIZE = 2048
MINI_BATCH = 64
STATE_DIM_PER_USER = 9
SCENARIO_BLOCK_LEN = 800
tx_power_dbm = 30
noise_floor_dbm = -94
precoder_type = "zf"

# AE and Compression
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32] # candidate compression ratios
FINETUNE_STEPS = 1 # light online adaptation step count on selected AE
FINETUNE_LR = 1e-4
FT_BUFFER_MIN = 8           # minimum recent samples before online FT
FT_START_SLOT = 32          # delay FT until buffer/statistics warm up
FT_INTERVAL = 5             # run FT periodically, not every slot
FT_NMSE_TRIGGER = 0.03      # only FT when quality drifts above this NMSE
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CR_ACTIONS = len(COMPRESSION_RATIOS)       # 4: directly choose one CR ratio
N_FT_ACTIONS = 1                              # kept for compatibility; RL no longer chooses FT
TOTAL_ACTIONS = N_CR_ACTIONS

# Reward weights
ALPHA_NMSE = 2.4
BETA_BITS = 0.20       # latent-size communication cost, normalized by largest latent payload
GAMMA_THROUGHPUT = 0.45 # keep throughput important without making one mid-rate ratio dominate
FINETUNE_COST = 0.0    # online fine tuning is disabled for this experiment
EARLY_NONKEEP_BONUS = 0.0
REWARD_STD_PENALTY = 0.0   # disable noisy cross-user reward coupling by default
RL_DEBUG = True
RL_DEBUG_INTERVAL = 200
RL_DEBUG_WINDOW = 200
SWITCH_PENALTY = 0.005
BUDGET_PENALTY_SCALE = 6.0
LATENT_Q_BITS = 10.0          # bits per latent element
TARGET_BUDGET_RATIO = 1/8    # fallback reference only; no slot-budget penalty is applied unless dataset budgets exist

TRAIN_ALLOW_FINETUNE = False  # freeze env adaptation during RL policy training stage
USE_ONLINE_NORMALIZER = False # training default
# Keep eval normalization aligned with training by default; enabling online
# adaptation only at test time can shift the AE input distribution and inflate NMSE.
EVAL_USE_ONLINE_NORMALIZER = USE_ONLINE_NORMALIZER
DQN_LR = 1e-4
GRAD_UPDATES_PER_STEP = 1
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-3
TRAIN_RANDOM_START = False
TRAIN_EPISODE_LEN = 2 * SCENARIO_BLOCK_LEN
# Misc
SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# ---- reward diagnostics storage ----
reward_terms_log = {
    "rate": [],
    "nmse": [],
    "bits": [],
    "finetune": [],
    "switch": [],
    "budget": [],
    "total_reward": []
}
# ---------------------------- Utilities -----------------------------------
def preprocess_csi_with_truncation(CSI, L_DELAY):
    """
    CSI: (T, K, NumBSant, 1, Nsub) complex frequency-domain
    Returns:
        CSI_trunc: (T, K, NumBSant, 1, L_DELAY) complex delay-domain
    """
    # Frequency → delay
    # CSI_delay = np.fft.ifft(CSI, axis=-1)
    Nsub = CSI.shape[4]
    CSI_delay = np.fft.ifft(CSI, axis=-1) * np.sqrt(Nsub)
    
    # Truncate taps
    CSI_delay_trunc = CSI_delay[..., :L_DELAY]
    power = np.mean(np.abs(CSI_delay_trunc)**2)
    CSI_delay_trunc /= np.sqrt(power + 1e-12)
    return CSI_delay_trunc

# CSI preprocessing for Paper model baseline
def preprocess_csi_for_3dcnn(csi_complex):
    """
    csi_complex: numpy complex array [Nr, Nt, Nsub] (or similar)
    Returns: torch tensor [1, 2, D, H, W]
    """
    real = np.real(csi_complex)
    imag = np.imag(csi_complex)
    csi_ri = np.stack([real, imag], axis=0)  # [2, ...]
    return torch.tensor(csi_ri, dtype=torch.float32).unsqueeze(0)
# Loading dataset
def load_dataset(path):

    """Load MATLAB .mat dataset produced by the generator script.
    Expects fields: CSI (Nsub x 1 x NumBSant x K x numSlots)
    And metadata arrays: userPos, userSpeed, userSNR_dB, scenarioLabels
    """
    try:
        
        with h5py.File(path, 'r') as f:
            # Load CSI (Nsub x 1 x NumBSant x K x numSlots)
            CSI_struct = np.array(f['CSI'])
            CSI = CSI_struct['real'] + 1j * CSI_struct['imag']
            CSI_trunc = preprocess_csi_with_truncation(CSI, L_DELAY=64)
            # Permute to (numSlots x K x NumBSant x 1 x Nsub) for easier time/user iteration
            # MATLAB was (Nsub, 1, NumBSant, K, numSlots). New order (T, K, A, 1, S)
            # CSI_perm = np.transpose(CSI, (4, 3, 2, 1, 0)) 

            # Load metadata
            # Transpose to (numSlots x K x ...) to align with CSI time axis (T, K)
            user_snr_dB = np.array(f['userSNR_dB'])
            user_speed = np.array(f['userSpeed']) if 'userSpeed' in f else None
            scenario_labels = np.array(f['scenarioLabels'])
            slot_budget_bits = np.array(f['slotBudgetBits']).reshape(-1) if 'slotBudgetBits' in f else None
            #user_pos = np.transpose(np.array(f['userPos']), (0, 1, 2))

            data = {
                'CSI': CSI_trunc, # (T, K, NumBSant, 1, Nsub)
                'userSNR_dB': user_snr_dB, # (T, K)
                'userSpeed': user_speed, # optional (T, K)
                'slotBudgetBits': slot_budget_bits, # optional (T,)
                'scenarioLabels': scenario_labels # (T, K)
                #'userPos': user_pos # (T, K, 2)
            }
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}. Please ensure the dataset path is correct and the file is in HDF5 format (-v7.3).")
        return None

# Computing SINR with zero-forcing 
def compute_sinr_from_csi(
    H_true,          # shape: (K, NumBSant, Nsub) complex
    H_est,           # shape: (K, NumBSant, Nsub) complex
    tx_power_dbm=30,
    noise_floor_dbm=-94,
    precoder_type="zf"
):
    K, NumBSant, Nsub = H_true.shape
    Ptx = 10 ** ((tx_power_dbm - 30) / 10) / K
    noise = 10 ** ((noise_floor_dbm - 30) / 10)
    sinr_users = np.zeros(K, dtype=float)
    per_user_rate = np.zeros(K, dtype=float)
    total_rate = 0.0

    for n in range(Nsub):
        H_n_true = H_true[:, :, n] # (K, Ant)
        H_n_est = H_est[:, :, n]   # (K, Ant)
        if precoder_type == "zf":
            # Base Station computes Precoder W using ESTIMATED CSI (H_est)
            # Zero-Forcing: W = H_est^H * inv(H_est * H_est^H)
            try:
                W_n = np.linalg.pinv(H_n_est) # (Ant, K)
            except np.linalg.LinAlgError:
                W_n = np.zeros((NumBSant, K), dtype=complex)

            # Normalize columns
            for k in range(K):
                norm_w = np.linalg.norm(W_n[:, k])
                if norm_w > 1e-9:
                    W_n[:, k] = W_n[:, k] / norm_w
        elif precoder_type == "MRT":
            W_n = np.zeros((NumBSant, K), dtype=complex)
            for k in range(K):
                h_k = H_n_est[k, :]  # (Ant,)
                norm_h = np.linalg.norm(h_k)
                if norm_h > 1e-9:
                    W_n[:, k] = np.conj(h_k) / norm_h
        else:
            raise ValueError(f"Unknown precoder type: {precoder_type}")
        # Apply Precoder to TRUE Channel to see real Interference
        # Received signal Y = H_true * W * x + n
        # Effective Channel Matrix H_eff = H_true @ W_n  -> Shape (K, K)
        # H_eff[k, j] is the signal received by User k from Beam j 
        H_eff = H_n_true @ W_n   # (K, K)     
        # Accumulate per-user SINR (linear) and sum-rate per subcarrier
        for u in range(K):
            signal = Ptx * np.abs(H_eff[u, u])**2
            interf = np.sum(Ptx * np.abs(H_eff[u, :])**2) - signal
            sinr = signal / (interf + noise + 1e-15)
            rate_u = np.log2(1.0 + sinr)
            sinr_users[u] += sinr
            per_user_rate[u] += rate_u
            total_rate += rate_u
    # average over subcarriers
    sinr_users /= Nsub
    per_user_rate /= Nsub
    total_rate /= Nsub
    return sinr_users, per_user_rate, total_rate

def calculate_normalization_stats(data):
    """Calculates mean and std for the real/imaginary flattened CSI."""
    CSI_data = data['CSI']
    T, K = CSI_data.shape[0], CSI_data.shape[1]
    def flatten_real_imag(H):
        H_flat = H.reshape(-1)
        power_scale = np.sqrt(np.mean(np.abs(H_flat) ** 2) + 1e-12)
        H_scaled = H_flat / power_scale
        return np.concatenate([np.real(H_scaled), np.imag(H_scaled)]).astype(np.float32)
    
    print("Calculating normalization statistics...")
    MU, SIGMA = [], []
    for k in range(K):
        user_samples = np.stack([flatten_real_imag(CSI_data[t, k, ...]) for t in range(T)])  # (T, feature_dim)
        mu_k = np.mean(user_samples, axis=0)
        sigma_k = np.std(user_samples, axis=0) + 1e-8
        MU.append(torch.tensor(mu_k, dtype=torch.float32).to(DEVICE))
        SIGMA.append(torch.tensor(sigma_k, dtype=torch.float32).to(DEVICE))
    
    return MU, SIGMA

def compute_nmse(recon, target):
    # both complex numpy arrays
    num = np.sum(np.abs(target - recon)**2)
    den = np.sum(np.abs(target)**2) + 1e-12
    return num/den

def flatten_real_imag_with_power(H):
    """Return power-normalized real/imag features and the sample RMS scale."""
    H_flat = H.reshape(-1)
    power_scale = np.sqrt(np.mean(np.abs(H_flat) ** 2) + 1e-12).astype(np.float32)
    H_scaled = H_flat / power_scale
    flat_ri = np.concatenate([np.real(H_scaled), np.imag(H_scaled)]).astype(np.float32)
    return flat_ri, power_scale

def restore_complex_from_flat(flat_ri, power_scale):
    """Rebuild a complex vector from real/imag features and restore sample gain."""
    n_half = len(flat_ri) // 2
    H_scaled = flat_ri[:n_half] + 1j * flat_ri[n_half:]
    return H_scaled * power_scale

def throughput_proxy(sinr_db, nmse_val, rate):
    
    nmse_clamped = min(max(nmse_val, 0.0), 1.0)
    return rate * (1.0 - nmse_clamped)

# Compute latent dim from compression ratio
def latent_dim_from_ratio(full_dim, ratio):
    # compression ratio = latent_dim / full_dim (approx)
    ld = max(8, int(round(full_dim * ratio)))
    return ld
def compute_reward_terms(rate, nmse, bits, did_finetune, budget_penalty, switch_cost=0.0, system_sum_rate=None):
    nmse_penalty = np.log1p(nmse)
    ft_cost = FINETUNE_COST if did_finetune else 0.0
    bit_weight = BETA_BITS

    total_reward = (
        GAMMA_THROUGHPUT * rate
        - ALPHA_NMSE * nmse_penalty
        - bit_weight * bits
        - ft_cost
        - switch_cost
        - budget_penalty
    )

    return {
        "rate": rate,
        "nmse": nmse,
        "bits": bits,
        "bit_weight": bit_weight,
        "finetune": ft_cost,
        "switch": switch_cost,
        "budget": budget_penalty,
        "system_sum_rate": np.nan if system_sum_rate is None else system_sum_rate,
        "total_reward": total_reward
    }

def build_joint_state_for_user(state_list, user_idx):
    """
    Build a user-specific full joint state by placing the current user's local
    state first, followed by the other users' states in a fixed order.
    """
    ordered = [state_list[user_idx]]
    ordered.extend(state_list[j] for j in range(len(state_list)) if j != user_idx)
    return np.concatenate(ordered, axis=0).astype(np.float32)

class RewardLogger:
    def __init__(self, name="default"):
        self.name = name
        self.data = {
            "rate": [],
            "nmse": [],
            "bits": [],
            "bit_weight": [],
            "finetune": [],
            "switch": [],
            "budget": [],
            "system_sum_rate": [],
            "total_reward": []
        }

    def log(self, reward_info):
        for k in reward_info:
            for key in self.data:
                self.data[key].append(reward_info[k][key])

    def save(self, save_dir):
        path = os.path.join(save_dir, f"{self.name}.mat")
        savemat(path, {k: np.array(v) for k, v in self.data.items()})
        print(f"[Logger] Saved: {path}")
# ---------------------------- Autoencoder -------------------------------
class SimpleAE(nn.Module):
    """Simple fully-connected autoencoder that operates on flattened real/imag CSI."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        layers = []
        dims = [input_dim]
        dim = input_dim
        
        while dim > latent_dim * 2:
            next_dim = dim // 2
            layers.append(nn.Linear(dim, next_dim))
            layers.append(nn.LayerNorm(next_dim))
            layers.append(nn.ReLU())
            dim = next_dim

        layers.append(nn.Linear(dim, latent_dim))
        dims.append(latent_dim)
        self.encoder = nn.Sequential(*layers)

        layers = []
        reversed_dims = dims[::-1]  # reverse list
        for i in range(len(reversed_dims) - 1):
            in_dim = reversed_dims[i]
            out_dim = reversed_dims[i + 1]
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            # No activation on last layer
            if i < len(reversed_dims) - 2:
                layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        self.value_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.feature(x)
        V = self.value_head(h)
        A = self.adv_head(h)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q
    
    def act(self, x):
        h = self.feature(x)
        A = self.adv_head(h)
        return A
class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.long, device=DEVICE),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buffer)
# ---------------------- ATH Rule-Based Controller -------------------------

def pretrain_ae_per_ratio(data, compression_ratios, save_dir, epochs=100, batch_size=512, mu_sigma=None):
    os.makedirs(save_dir, exist_ok=True)

    CSI_data = data['CSI']
    T, K = CSI_data.shape[0], CSI_data.shape[1]

    # Flatten all CSI samples. When per-user normalization stats are available,
    # apply the same normalization used later during evaluation so the AE sees
    # a consistent input distribution at train and test time.
    print("[Pretrain] Preparing dataset...")
    samples = []
    sample_user_indices = []
    for t in range(T):
        for k in range(K):
            H = CSI_data[t, k, ...]
            H_flat, _ = flatten_real_imag_with_power(H)
            samples.append(H_flat)
            sample_user_indices.append(k)

    X = np.array(samples, dtype=np.float32)
    input_dim = X.shape[1]

    if mu_sigma is not None:
        MU, SIGMA = mu_sigma
        X_norm = np.empty_like(X, dtype=np.float32)
        for idx, k in enumerate(sample_user_indices):
            mu_k = MU[k].detach().cpu().numpy().astype(np.float32)
            sigma_k = SIGMA[k].detach().cpu().numpy().astype(np.float32)
            X_norm[idx] = (X[idx] - mu_k) / (sigma_k + 1e-8)
    else:
        # Fallback path retained for compatibility, but per-user stats are preferred.
        mu = np.mean(X, axis=0, keepdims=True).astype(np.float32)
        sigma = np.std(X, axis=0, keepdims=True).astype(np.float32)
        sigma = np.clip(sigma, 1e-8, None)
        X_norm = (X - mu) / sigma

    X = torch.tensor(X_norm, dtype=torch.float32).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    checkpoint_paths = {}

    for ratio in compression_ratios:
        # epochs = int(50 * (0.25 / ratio))
        print(f"\n[Pretrain] Training AE for CR={ratio}")

        latent_dim = latent_dim_from_ratio(input_dim, ratio)
        model = SimpleAE(input_dim, latent_dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for ep in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                recon, _ = model(batch)
                loss = F.mse_loss(recon, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"  Epoch {ep+1}: Loss={total_loss/len(loader):.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"ae_cr_{int(1/ratio)}.pt")
        torch.save({
            "state_dict": model.state_dict(),
            "normalization_mode": "per_user_runtime" if mu_sigma is not None else "global_fallback"
        }, ckpt_path)

        checkpoint_paths[ratio] = ckpt_path
        print(f"[Saved] {ckpt_path}")

    return checkpoint_paths
# ---------------------------- Multi-User Environment -------------------------------
class MultiUserCSIEnv:
    """Environment wrapper over the dataset, operating on all K users at each slot."""
    def __init__(self, data, compression_list=COMPRESSION_RATIOS, mu_sigma=None, base_checkpoint=None, allow_finetune=False,
                 use_online_normalizer=USE_ONLINE_NORMALIZER):
        self.data = data
        # CSI shape: (numSlots, K, NumBSant, 1, Nsub)
        self.CSI = data['CSI'] 
        self.numSlots = self.CSI.shape[0]
        self.K = self.CSI.shape[1] # Number of users
        self.NumBSant = self.CSI.shape[2]
        self.Nsub = self.CSI.shape[4]
        self.input_dim = 2 * self.Nsub * self.NumBSant # 2x for real/imag
        self.mu = mu_sigma[0] if mu_sigma is not None else None
        self.sigma = mu_sigma[1] if mu_sigma is not None else None
        self.base_checkpoint = base_checkpoint
        self.allow_finetune = allow_finetune
        self.use_online_normalizer = use_online_normalizer
        
        # Load metadata (indexed as [time_slot, user_idx])
        self.snr = data.get('userSNR_dB', None)
        self.sinr_db = np.zeros(self.K)
        # self.speed = data.get('userSpeed', None)
        self.scenarioLabels = data.get('scenarioLabels', None)
        self.slot_budget_bits = data.get('slotBudgetBits', None)
        self.compression_list = compression_list
        self.action_space_per_user = len(compression_list) # 0-3 directly choose one CR ratio
        self.action_space = self.K * self.action_space_per_user # Total output dim of the RL model 
        self.state_dim = self.K * STATE_DIM_PER_USER
        self.normalizer = OnlineNormalizer(
            K=self.K,
            feature_dim=self.input_dim,
            init_mu=mu_sigma[0] if mu_sigma else None,
            init_sigma=mu_sigma[1] if mu_sigma else None,
            momentum=0.01  # adapts over ~100 slots when enabled
        )
        # Running mean/std to normalize NMSE in 
        self.nmse_mean = np.zeros(self.K)
        self.nmse_var = np.ones(self.K)
        self.nmse_count = np.ones(self.K) * 1e-4
        # Internal tracking (lists of K elements)
        self.t = 0
        self.last_ratio = [compression_list[0]] * self.K
        self.last_nmse = [0.0] * self.K
        self.time_since_finetune = [0.0] * self.K
        self.total_finetunes = [0] * self.K
        self.last_finetune_loss = [0.0] * self.K
        
        self.ae_latent_dim = [
            latent_dim_from_ratio(self.input_dim, self.compression_list[0])
        ] * self.K
        self.max_bits_per_user = self._payload_bits(max(self.compression_list))  # largest latent payload
        self.max_total_bits = self.K * self._payload_bits(TARGET_BUDGET_RATIO)   # compatibility fallback only
        self.bit_weight_eff = BETA_BITS / max(1, self.K)
        # Keep a model bank per user per ratio.
        # Switching ratios then becomes a fast pointer swap instead of re-init/loading.
        self.ae_bank = [dict() for _ in range(self.K)]
        self.opt_bank = [dict() for _ in range(self.K)]
        self.aes = [None] * self.K
        self.ft_optimizers = [None] * self.K

        for k in range(self.K):
            for ratio in self.compression_list:
                ae, opt = self._init_ae(ratio)
                self.ae_bank[k][ratio] = ae
                self.opt_bank[k][ratio] = opt
            default_ratio = self.compression_list[0]
            self.aes[k] = self.ae_bank[k][default_ratio]
            self.ft_optimizers[k] = self.opt_bank[k][default_ratio]
        # Per user rolling buffer
        self.ft_buffers = [deque(maxlen=32) for _ in range(self.K)]

    def _update_nmse_stats(self, k, x):
        count = self.nmse_count[k]
        mean = self.nmse_mean[k]
        var = self.nmse_var[k]

        count_new = count + 1
        delta = x - mean
        mean_new = mean + delta / count_new
        var_new = var + delta * (x - mean_new)

        self.nmse_mean[k] = mean_new
        self.nmse_var[k] = var_new
        self.nmse_count[k] = count_new
    def _normalize_nmse(self, k, x):
        std = np.sqrt(self.nmse_var[k] / self.nmse_count[k])
        return np.clip((x - self.nmse_mean[k]) / (std + 1e-8), -5, 5)   
     
    def _init_ae(self, ratio):
        # print(f"[DEBUG] base_checkpoint type: {type(self.base_checkpoint)}")
        # print(f"[DEBUG] available keys: {list(self.base_checkpoint.keys()) if isinstance(self.base_checkpoint, dict) else 'N/A'}")
        # print(f"[DEBUG] requested ratio: {ratio}, type={type(ratio)}")
       
        ld = latent_dim_from_ratio(self.input_dim, ratio)
        new_ae = SimpleAE(self.input_dim, ld).to(DEVICE)
        
        ckpt_path = None
        if isinstance(self.base_checkpoint, dict):
            ckpt_path = self.base_checkpoint.get(ratio, None)

        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            try:
                new_ae.load_state_dict(ckpt["state_dict"])
                # print(f"[AE] Loaded FULL checkpoint for ratio={ratio}")
            except Exception as e:
                print(f"[AE] ERROR loading full model for ratio={ratio}: {e}")

        # else:
        #     print(f"[AE] WARNING: No checkpoint for ratio={ratio}")

        optimizer = torch.optim.Adam(new_ae.parameters(), lr=FINETUNE_LR)
        return new_ae, optimizer
    
    def _payload_bits(self, ratio):
        # Feedback payload scales with the AE latent dimension: each latent value is quantized
        # to LATENT_Q_BITS before being sent from the user to the BS.
        latent = latent_dim_from_ratio(self.input_dim, ratio)
        return LATENT_Q_BITS * latent

    def _bit_cost_norm(self, ratio):
        return self._payload_bits(ratio) / (self.max_bits_per_user + 1e-12)

    def _get_slot_budget_bits(self, t):
        if self.slot_budget_bits is None:
            return None
        if len(self.slot_budget_bits) == 0:
            return None
        budget = float(self.slot_budget_bits[min(t, len(self.slot_budget_bits) - 1)])
        return max(budget, 1e-12)

    def _compute_budget_penalty_per_user(self, slot_payload_bits, slot_idx):
        slot_budget_bits = self._get_slot_budget_bits(slot_idx)
        if slot_budget_bits is None:
            return 0.0
        budget_mismatch = abs(slot_payload_bits - slot_budget_bits) / slot_budget_bits
        return BUDGET_PENALTY_SCALE * budget_mismatch / max(1, self.K)
    
    def _get_csi_slot(self, t):
        # CSI stored as: (T, K, NumBSant, 1, Nsub)
        H = self.CSI[t, :, :, 0, :]    # K × NumBSant × Nsub    
        # H = np.transpose(H, (2, 1, 0))     
        return H
    
    def _refresh_observation(self, slot_idx, update_normalizer, update_buffers):
        """Build a consistent pre-action observation for one slot."""
        H_slot = self._get_csi_slot(slot_idx)
        raw_csi_batch = self._get_raw_csi_for_all_users(slot_idx)
        H_est_slot = np.zeros_like(H_slot, dtype=complex)

        for k in range(self.K):
            x_flat, power_scale = self._flatten_real_imag_with_power(raw_csi_batch[k])
            mu_k, sigma_k = self.normalizer.get_tensors(k, DEVICE)
            x_t = torch.from_numpy(x_flat).float().unsqueeze(0).to(DEVICE)
            x_norm = (x_t - mu_k) / sigma_k

            if update_normalizer:
                self.normalizer.update(k, x_flat)
            if update_buffers:
                self.ft_buffers[k].append(x_norm.detach())

            with torch.no_grad():
                recon_norm, _ = self.aes[k](x_norm)

            recon_norm_np = recon_norm.cpu().numpy().flatten()
            mu_np = mu_k.cpu().numpy()
            sigma_np = sigma_k.cpu().numpy()
            recon_raw = recon_norm_np * sigma_np + mu_np
            recon_c = restore_complex_from_flat(recon_raw, power_scale)
            target_c = restore_complex_from_flat(x_flat, power_scale)

            self.last_nmse[k] = compute_nmse(recon_c, target_c)
            self._update_nmse_stats(k, self.last_nmse[k])
            H_est_slot[k] = recon_c.reshape(H_slot.shape[1], H_slot.shape[2])

        self.H_est = H_est_slot
        self.sinr_db, _, _ = compute_sinr_from_csi(
            H_slot,
            self.H_est,
            tx_power_dbm=tx_power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            precoder_type=precoder_type
        )
        return [self._get_user_state(k) for k in range(self.K)]
    def reset(self, start_idx=0):
        self.t = int(np.clip(start_idx, 0, max(0, self.numSlots - 1)))
        self.normalizer.reset_to_initial_stats()
        self.last_ratio = [self.compression_list[0]] * self.K
        self.nmse_mean = np.zeros(self.K)
        self.nmse_var = np.ones(self.K)
        self.nmse_count = np.ones(self.K) * 1e-4
        self.last_nmse = [0.0] * self.K
        self.time_since_finetune = [0.0] * self.K
        self.total_finetunes = [0] * self.K
        self.last_finetune_loss = [0.0] * self.K
        self.ft_buffers = [deque(maxlen=32) for _ in range(self.K)]

        # Re-initialize per-user, per-ratio AE banks before initial observation
        # so reset starts from pretrained checkpoints consistently.
        for k in range(self.K):
            for ratio in self.compression_list:
                ae, opt = self._init_ae(ratio)
                self.ae_bank[k][ratio] = ae
                self.opt_bank[k][ratio] = opt
            default_ratio = self.compression_list[0]
            self.aes[k] = self.ae_bank[k][default_ratio]
            self.ft_optimizers[k] = self.opt_bank[k][default_ratio]

        return self._refresh_observation(
            slot_idx=self.t,
            update_normalizer=self.use_online_normalizer,
            update_buffers=False
        )

    def _get_raw_csi_for_all_users(self, idx):
        # Returns (K x NumBSant x 1 x Nsub) complex array
        return self.CSI[idx, :, :, :, :] 

    def _flatten_real_imag(self, H):
        # H: NumBSant x 1 x Nsub complex
        # Seperate real and imaginary parts
        H_flat = H.reshape(-1)
        re = np.real(H_flat)
        im = np.imag(H_flat)
        return np.concatenate([re, im]).astype(np.float32)

    def _flatten_real_imag_with_power(self, H):
        return flatten_real_imag_with_power(H)
    
    def _get_user_state(self, k):
        sinr_db_k = 10.0 * np.log10(self.sinr_db[k] + 1e-9)
        sinr_norm_k = float(np.clip(sinr_db_k / 30.0, 0.0, 1.0))
        nmse_norm = self._normalize_nmse(k, self.last_nmse[k])
        if self.scenarioLabels is not None:
            scenario_raw = float(self.scenarioLabels[min(self.t, self.numSlots - 1), k])
            scenario_norm = float(np.clip((scenario_raw - 1.0) / 2.0, 0.0, 1.0))
        else:
            scenario_norm = 0.0
        phase_in_block = float((self.t % SCENARIO_BLOCK_LEN) / max(1, SCENARIO_BLOCK_LEN - 1))
        # ----- correlation between users -----
        H_k = self.H_est[k, :, 0] 
        correlations = []
        for j in range(self.K):
            if j != k:
                H_j = self.H_est[j, :, 0]
                corr = np.abs(np.vdot(H_k, H_j)) / ((np.linalg.norm(H_k) * np.linalg.norm(H_j)) + 1e-12)
                correlations.append(corr)
        
        avg_corr = np.mean(correlations) if correlations else 0.0
        current_others_usage = sum(self._payload_bits(r) for i, r in enumerate(self.last_ratio) if i != k)
        slot_budget_bits = self._get_slot_budget_bits(self.t)
        if slot_budget_bits is None:
            budget_headroom = 0.5
        else:
            budget_headroom = (slot_budget_bits - current_others_usage) / (slot_budget_bits + 1e-12)
        ft_age_feature = (self.time_since_finetune[k] / 100.0 if self.allow_finetune else 0.0)
        
        state_k = np.array([
            self.last_ratio[k], 
            nmse_norm, 
            self.last_nmse[k], #np.log1p(self.last_nmse[k]),
            sinr_norm_k, 
            avg_corr,
            budget_headroom,
            ft_age_feature,
            scenario_norm,
            phase_in_block
        ], dtype=np.float32)

        return state_k

    def step(self, action_vector):
        
        done = False
        idx = self.t
        rawH_k = self._get_raw_csi_for_all_users(idx) # (K, NumBSant, 1, Nsub) complex
        H_true_slot = self._get_csi_slot(self.t)
        H_est_rl = np.zeros_like(H_true_slot, dtype=complex)
        rewards = np.zeros(self.K, dtype=float)
        bits_this_slot = np.zeros(self.K, dtype=float)
        did_finetune = np.zeros(self.K, dtype=bool)
        switched = np.zeros(self.K, dtype=bool)
        slot_payload_bits = 0.0
        reward_info = {}
        for k in range(self.K):
            cr_action = int(action_vector[k])   # 0-3 = directly select a compression ratio
            cr_action = int(np.clip(cr_action, 0, len(self.compression_list) - 1))
            
            ae = self.aes[k]
            ft_optimizer = self.ft_optimizers[k]
            
            x, power_scale = self._flatten_real_imag_with_power(rawH_k[k, ...])
            mu_k, sigma_k = self.normalizer.get_tensors(k, DEVICE)

            x_t = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
            x_t_norm = (x_t - mu_k) / sigma_k
            if self.use_online_normalizer:
                self.normalizer.update(k, x)
            self.ft_buffers[k].append(x_t_norm.detach())
            # --- Step 1: CR selection (cheap pointer swap if the chosen ratio changes) ---
            ratio_chosen = self.compression_list[cr_action]
            if ratio_chosen != self.last_ratio[k]:
                switched[k] = True
                self.aes[k] = self.ae_bank[k][ratio_chosen]
                self.ft_optimizers[k] = self.opt_bank[k][ratio_chosen]
                ae = self.aes[k]
                ft_optimizer = self.ft_optimizers[k]

            self.last_ratio[k] = ratio_chosen
            self.ae_latent_dim[k] = ae.latent_dim

            # Online fine tuning is skipped unless allow_finetune is explicitly enabled.
            do_finetune = (
                self.allow_finetune
                and ft_optimizer is not None
                and len(self.ft_buffers[k]) >= FT_BUFFER_MIN
                and self.t >= FT_START_SLOT
                and (
                    switched[k]
                    or ((self.t % FT_INTERVAL == 0) and (self.last_nmse[k] > FT_NMSE_TRIGGER))
                )
            )
            if do_finetune:
                _ = self._finetune_on_sample(
                    x_t_norm,
                    ae,
                    ft_optimizer,
                    buffer=self.ft_buffers[k]
                )
                did_finetune[k] = True
                self.time_since_finetune[k] = 0.0
                self.total_finetunes[k] += 1
            elif self.allow_finetune:
                self.time_since_finetune[k] += 1.0
            else:
                self.time_since_finetune[k] = 0.0
    
            # --- Forward pass with current AE ---
            with torch.no_grad():
                recon_norm, _ = ae(x_t_norm)

            recon_norm = recon_norm.cpu().numpy().flatten()
            # NMSE in raw space (v5-style) to reduce non-stationary reward targets.
            mu_np = mu_k.cpu().numpy()
            sigma_np = sigma_k.cpu().numpy()
            recon_raw = recon_norm * sigma_np + mu_np
            recon_complex = restore_complex_from_flat(recon_raw, power_scale)
            target_complex = restore_complex_from_flat(x, power_scale)
            self.last_nmse[k] = compute_nmse(recon_complex, target_complex)
            self._update_nmse_stats(k, self.last_nmse[k])
            H_est_rl[k, :, :] = recon_complex.reshape(H_true_slot.shape[1], H_true_slot.shape[2])
            
            bits_this_slot[k] = self._bit_cost_norm(self.last_ratio[k])   # normalized per-user bit cost
            slot_payload_bits += self._payload_bits(self.last_ratio[k])   # absolute payload for budget
         
        # Compute per-user SINR with all users' reconstructions ---
        sinr_users, per_user_rates, system_sum_rate = compute_sinr_from_csi(
            H_true_slot,
            H_est_rl,
            tx_power_dbm=tx_power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            precoder_type="zf"
        )
        self.sinr_db = sinr_users
        
        budget_penalty_per_user = self._compute_budget_penalty_per_user(slot_payload_bits, idx)
        
        # Reward Calculation
        rewards = np.zeros(self.K, dtype=float)
        for k in range(self.K):
            # Exact average user rate over subcarriers.
            rate_k = per_user_rates[k]
            nmse_penalty = np.log1p(self.last_nmse[k])
            switch_cost = SWITCH_PENALTY if switched[k] else 0.0
            ft_cost = FINETUNE_COST if did_finetune[k] else 0.0

            rewards[k] = (
                GAMMA_THROUGHPUT * rate_k 
                - ALPHA_NMSE * nmse_penalty
                - self.bit_weight_eff * bits_this_slot[k] 
                - switch_cost
                - ft_cost
                - budget_penalty_per_user
            )
            reward_info[k] = {
                "rate": rate_k,
                "nmse": self.last_nmse[k],
                "bits": bits_this_slot[k],
                "bit_weight": self.bit_weight_eff,
                "finetune": ft_cost,
                "switch": switch_cost,
                "budget": budget_penalty_per_user,
                "system_sum_rate": system_sum_rate,
                "total_reward": rewards[k]
            }

        self.H_est = H_est_rl

        # advance time first, then build next_state as before
        self.t += 1
        if self.t >= self.numSlots-1:
            done = True
        scenario_boundary = False
        if self.scenarioLabels is not None and not done:
            prev_labels = self.scenarioLabels[idx, :]
            next_labels = self.scenarioLabels[self.t, :]
            scenario_boundary = bool(np.any(prev_labels != next_labels))

        next_state = None
        if not done:
            next_state = self._refresh_observation(
                slot_idx=self.t,
                update_normalizer=self.use_online_normalizer,
                update_buffers=False
            )
        else:
            next_state = [self._get_user_state(k) for k in range(self.K)]
        
        # The total reward is the sum of rewards for all K users
        return next_state, rewards, done, {
            "reward_info": reward_info,
            "system_sum_rate": system_sum_rate,
            "scenario_boundary": scenario_boundary
        }

    def _finetune_on_sample(self, x_t, ae, ft_optimizer, buffer=None):
        # perform FINETUNE_STEPS gradient updates on AE with MSE loss using x_t
        ae.train()
        if buffer is not None and len(buffer) >= 4:
            batch = torch.cat(list(buffer), dim=0)  # (N, feature_dim)
        else:
            batch = x_t  # fallback to single sample only when buffer too small
        for _ in range(FINETUNE_STEPS):
            ft_optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = nn.MSELoss()(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            ft_optimizer.step()
            last_loss = loss.item()
        ae.eval()
        return last_loss

# ---------------------------- RL Agent ----------------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=200_000):
        self.q_net = DuelingQNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.gamma = 0.95
        self.batch_size = 256
        self.learning_starts = 500
        self.update_steps = 0
        self.target_update_freq = 1000
        self.epsilon = 0.8
        self.epsilon_min = 0.01 # 0.05
        self.epsilon_decay = 0.9998

    @staticmethod
    def _sanitize_state(state):
        return np.nan_to_num(
            np.asarray(state, dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )

    def select_action(self, state):
        """Training-time action (epsilon-greedy)"""
        state = self._sanitize_state(state)
        if random.random() < self.epsilon:
            return random.randrange(self.q_net.adv_head.out_features)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return self.q_net(s).argmax(dim=1).item()
    
    def select_greedy_action(self, state):
        """Evaluation-time action: strictly greedy (epsilon=0)."""
        state = self._sanitize_state(state)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return self.q_net(s).argmax(dim=1).item()

    def store(self, s, a, r, s_next, done):
        s = self._sanitize_state(s)
        s_next = self._sanitize_state(s_next)
        r = float(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))
        self.replay.add(s, a, r, s_next, done)

    def train_step(self):
        if len(self.replay) < self.learning_starts:
            return None
        s, a, r, s_next, d = self.replay.sample(self.batch_size)
        s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0)
        s_next = torch.nan_to_num(s_next, nan=0.0, posinf=1.0, neginf=-1.0)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_net(s_next).argmax(dim=1)
            next_q = self.target_net(s_next).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * next_q * (1 - d)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.SmoothL1Loss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
class ActorCritic(nn.Module):
    """
    The Actor now outputs K logits groups (K * action_dim_per_user).
    The Critic still outputs a single value for the whole state.
    """
    def __init__(self, state_dim, action_dim_per_user, K, hidden=128):
        super().__init__()
        self.K = K
        self.action_dim_per_user = action_dim_per_user
        
        # Actor outputs K * action_dim_per_user logits
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim_per_user)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

# Storage for trajectories
Transition = namedtuple('Transition', ['state','action','logp','reward','done','value'])


def evaluate_paper_adaptive_baseline(data, models, mu, sigma, compression_list, cr_classifier):
    
    print("\n--- Evaluating Paper Adaptive Baseline ---")
    K = data['CSI'].shape[1]
    T = data['CSI'].shape[0]
    
    env = MultiUserCSIEnv(
        data,
        mu_sigma=(mu, sigma),
        base_checkpoint=pretrain_paths,
        allow_finetune=False,
        use_online_normalizer=EVAL_USE_ONLINE_NORMALIZER
    )
    env.normalizer.reset_counts()
    rewards_hist = []
    actions_hist = [] 
    logger = RewardLogger(name="paper_baseline")
    last_ratio = [compression_list[0]] * K
    # NMSE Threshold for "Acceptable Quality" (Paper logic)
    # If NMSE < THRESHOLD with low CR, pick low CR. Else pick high CR.
    NMSE_THRESHOLD = 0.1 
    
    with torch.no_grad():
        for t in range(T):
            if t % 100 == 0: print(f"Evaluating Slot {t}/{T}")
            
            H_slot = env._get_raw_csi_for_all_users(t) # K x Ant x 1 x Sub
            H_true_slot = env._get_csi_slot(t) 
            H_est_paper = np.zeros_like(H_true_slot, dtype=complex)
            
            slot_rewards = []
            slot_actions = []
            final_nmse_per_user = {}
            final_cr_per_user   = {}
            for k in range(K):
                # Prepare Input
                rawH = H_slot[k, ...]
                x_flat, power_scale = env._flatten_real_imag_with_power(rawH)
                # --- Online normalization ---
                mu_k, sigma_k = env.normalizer.get_tensors(k, DEVICE)
                x_t = torch.from_numpy(x_flat).float().unsqueeze(0).to(DEVICE)
                x_norm = (x_t - mu_k) / sigma_k
                if env.use_online_normalizer:
                    env.normalizer.update(k, x_flat)
                # --- Adaptive Selection Logic ---
                with torch.no_grad():
                    cls_logits = cr_classifier(x_t)   # classifier sees CSI
                    cls_idx = int(torch.argmax(cls_logits, dim=1).item())
                    final_cr = compression_list[cls_idx]
                
                model = models[final_cr]
                model.eval()
                recon_n, _ = model(x_norm)
                
                recon_n_np = recon_n.cpu().numpy().flatten()
                mu_np = mu_k.cpu().numpy()
                sigma_np = sigma_k.cpu().numpy()

                recon_raw = recon_n_np * sigma_np + mu_np
                recon_c = restore_complex_from_flat(recon_raw, power_scale)
                target_c = restore_complex_from_flat(x_flat, power_scale)
                final_nmse = compute_nmse(recon_c, target_c)
                H_est_paper[k, :, :] = recon_c.reshape(H_true_slot.shape[1], H_true_slot.shape[2])

                final_nmse_per_user[k] = final_nmse
                final_cr_per_user[k] = final_cr
                
            sinr_users, per_user_rates, system_sum_rate = compute_sinr_from_csi(
                H_true_slot,
                H_est_paper,
                precoder_type="zf"
            )
            slot_payload_bits = sum(env._payload_bits(final_cr_per_user[k]) for k in range(K))
            budget_penalty_per_user = env._compute_budget_penalty_per_user(slot_payload_bits, env.t)

            for k in range(K):
                rate_k = per_user_rates[k]
                cr_k     = final_cr_per_user[k]
                nmse_log = np.log1p(final_nmse_per_user[k])
                bit_cost_k = env._bit_cost_norm(cr_k)
                switch_cost = SWITCH_PENALTY if cr_k != last_ratio[k] else 0.0
                reward   = (
                    GAMMA_THROUGHPUT * rate_k
                    - ALPHA_NMSE    * nmse_log
                    - env.bit_weight_eff * bit_cost_k
                    - switch_cost
                    - budget_penalty_per_user
                )
                slot_rewards.append(reward)
                reward_dict = {
                    "rate": rate_k,
                    "nmse": final_nmse_per_user[k],
                    "bits": bit_cost_k,
                    "bit_weight": env.bit_weight_eff,
                    "finetune": 0.0,
                    "switch": switch_cost,
                    "budget": budget_penalty_per_user,
                    "system_sum_rate": system_sum_rate,
                    "total_reward": reward
                }
                logger.log({k: reward_dict})
                try:
                    act_idx = compression_list.index(cr_k)
                except ValueError:
                    act_idx = 0
                slot_actions.append(act_idx)
                last_ratio[k] = cr_k
                
            rewards_hist.append(slot_rewards)
            actions_hist.append(slot_actions)
    logger.save(SAVE_DIR)        
    return np.array(rewards_hist), np.array(actions_hist)

# --------------------------- Training loop -------------------------------
def plot_rl_performance(reward_hist, scenario_hist, action_hist):
    # Convert to NumPy for easier indexing
    rewards = np.array(reward_hist)
    scenarios = np.array(scenario_hist)
    actions = np.array(action_hist)
    slots = np.arange(rewards.shape[0])
    
    plt.figure(figsize=(15, 12), facecolor='white')
    sns.set_style("whitegrid")

    # --- Plot 1: Reward Evolution (Smoothed) ---
    plt.subplot(2, 1, 1)
    for k in range(rewards.shape[1]):
        # Apply a rolling mean to see the trend through the noise
        smoothed_r = pd.Series(rewards[:, k]).rolling(window=200).mean()
        plt.plot(slots, smoothed_r, label=f'User {k} Reward (Smooth)')
    
    plt.title('Reward Evolution per User (Rolling Mean)', fontsize=14)
    plt.ylabel('Scalar Reward')
    plt.legend()

    # --- Plot 2: Reward Distribution per Scenario ---
    # Reshape data for Seaborn (Long Format)
    # df = pd.DataFrame({
    #     'Reward': rewards.flatten(),
    #     'Scenario': scenarios.flatten().astype(int),
    #     'User': np.tile(np.arange(rewards.shape[1]), rewards.shape[0])
    # })
    # # Map scenario IDs to names
    # scen_map = {1: 'LoS', 2: 'NLoS', 3: 'Blockage'}
    # df['Environment'] = df['Scenario'].map(scen_map)

    # plt.subplot(3, 1, 2)
    # sns.boxplot(data=df, x='Environment', y='Reward', hue='User', palette='Set2')
    # plt.title('Reward Distribution by Environment Type', fontsize=14)

    # --- Plot 3: Actions vs. Scenarios (The "Strategy" Heatmap) ---
    plt.subplot(2, 1, 2)
    # We want to see how often each action is taken in each scenario
    pivot_df = pd.DataFrame({
        'Action': actions.flatten(),
        'Scenario': scenarios.flatten().astype(int)
    })
    # Create a cross-tabulation (counts of action per scenario)
    action_counts = pd.crosstab(pivot_df['Scenario'], pivot_df['Action'], normalize='index')
    sns.heatmap(action_counts, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Action Frequency per Scenario (Strategy Heatmap)', fontsize=14)
    plt.xlabel('Action (0: CR=1/4, 1: CR=1/8, 2: CR=1/16, 3: CR=1/32)')
    plt.ylabel('Scenario (1:LoS, 2:NLoS, 3:Blockage)')

    plt.tight_layout()
    plt.savefig('RL_Performance_Analysis.png', dpi=300)
    plt.show()

def evaluate_greedy_policy(agent, data, mu_sigma, base_checkpoint):
    """Run one full epsilon=0 evaluation sweep and return slot-level histories."""
    K = data['CSI'].shape[1]
    env = MultiUserCSIEnv(
        data,
        mu_sigma=mu_sigma,
        base_checkpoint=base_checkpoint,
        allow_finetune=False,
        # Match the final evaluation environment so checkpoint selection
        # reflects the same observation dynamics used at test time.
        use_online_normalizer=EVAL_USE_ONLINE_NORMALIZER
    )
    state = env.reset()
    reward_history = []
    action_history = []
    scenario_history = []

    while True:
        actions = np.zeros(K, dtype=int)
        for k in range(K):
            state_k = build_joint_state_for_user(state, k)
            actions[k] = agent.select_greedy_action(state_k)

        next_state, rewards, done, _ = env.step(actions)
        reward_history.append(rewards.copy())
        action_history.append(actions.copy())
        scenario_history.append([env.scenarioLabels[env.t - 1, k] for k in range(env.K)])
        state = next_state

        if done:
            break

    return (
        np.array(reward_history),
        np.array(action_history),
        np.array(scenario_history)
    )

def _ema_smooth(series, alpha=0.35):
    """Simple EMA smoothing to make epoch-level trends easier to read."""
    series = np.asarray(series, dtype=np.float32)
    if len(series) == 0:
        return series

    smoothed = np.empty_like(series)
    smoothed[0] = series[0]
    for idx in range(1, len(series)):
        smoothed[idx] = alpha * series[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def plot_epoch_reward_summary(
    train_epoch_rewards,
    greedy_eval_epoch_rewards,
    save_path,
    best_epoch=None,
    best_greedy_eval_mean=None
):
    """Plot convergence with clearer aggregate trends and reduced per-user clutter."""
    train_epoch_rewards = np.asarray(train_epoch_rewards, dtype=np.float32)
    greedy_eval_epoch_rewards = np.asarray(greedy_eval_epoch_rewards, dtype=np.float32)
    if train_epoch_rewards.ndim == 1:
        train_epoch_rewards = train_epoch_rewards[:, None]
    if greedy_eval_epoch_rewards.ndim == 1:
        greedy_eval_epoch_rewards = greedy_eval_epoch_rewards[:, None]
    epochs = np.arange(1, train_epoch_rewards.shape[0] + 1)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')

    train_mean = np.mean(train_epoch_rewards, axis=1)
    eval_mean = np.mean(greedy_eval_epoch_rewards, axis=1)
    train_std = np.std(train_epoch_rewards, axis=1)
    eval_std = np.std(greedy_eval_epoch_rewards, axis=1)
    train_smooth = _ema_smooth(train_mean)
    eval_smooth = _ema_smooth(eval_mean)
    eval_gap = train_mean - eval_mean
    delta_from_start = eval_mean - eval_mean[0]

    ax = axes[0, 0]
    ax.plot(epochs, train_mean, marker='o', linewidth=2.0, color='tab:blue', alpha=0.35, label='Train Mean')
    ax.plot(epochs, eval_mean, marker='s', linewidth=2.0, color='tab:orange', alpha=0.35, label='Greedy Eval Mean')
    ax.plot(epochs, train_smooth, linewidth=2.8, color='tab:blue', label='Train Mean (EMA)')
    ax.plot(epochs, eval_smooth, linewidth=2.8, color='tab:orange', linestyle='--', label='Greedy Eval Mean (EMA)')
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='tab:blue', alpha=0.12)
    ax.fill_between(epochs, eval_mean - eval_std, eval_mean + eval_std, color='tab:orange', alpha=0.12)
    if best_epoch is not None and 1 <= best_epoch <= len(epochs):
        ax.axvline(best_epoch, color='gray', linestyle=':', linewidth=1.5)
        if best_greedy_eval_mean is not None:
            ax.scatter([best_epoch], [best_greedy_eval_mean], color='crimson', s=80, zorder=5, label='Best Greedy Eval')
            ax.annotate(
                f'Best epoch {best_epoch}\nEval={best_greedy_eval_mean:.3f}',
                xy=(best_epoch, best_greedy_eval_mean),
                xytext=(8, 10),
                textcoords='offset points',
                fontsize=10,
                color='crimson'
            )
    ax.set_title('Overall Convergence Trend', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Reward')
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    im = ax.imshow(
        greedy_eval_epoch_rewards.T,
        aspect='auto',
        cmap='YlGnBu',
        interpolation='nearest'
    )
    ax.set_title('Greedy Eval Reward by User and Epoch', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('User Index')
    ax.set_xticks(np.arange(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_yticks(np.arange(greedy_eval_epoch_rewards.shape[1]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Avg Reward')

    ax = axes[1, 0]
    ax.plot(epochs, eval_gap, marker='d', linewidth=2.2, color='tab:purple')
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.2)
    ax.set_title('Train - Greedy Eval Gap', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward Gap')

    ax = axes[1, 1]
    colors = ['tab:green' if x >= 0 else 'tab:red' for x in delta_from_start]
    ax.bar(epochs, delta_from_start, color=colors, alpha=0.85)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.2)
    ax.set_title('Greedy Eval Improvement vs. Epoch 1', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Delta Reward')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def train_agent_on_dataset(data, mu_sigma, base_checkpoint):
    # K is the second dimension of CSI (T, K, ...)
    K = data['CSI'].shape[1] 
    state_dim = STATE_DIM_PER_USER * K
    action_dim = TOTAL_ACTIONS
    if K <= 2:
        buffer_capacity = 200_000
    elif K <= 4:
        buffer_capacity = 300_000
    elif K <= 6:
        buffer_capacity = 400_000
    else:
        buffer_capacity = 500_000
    
    print(f"Replay buffer capacity for K={K}: {buffer_capacity:,}")
    agent = DQNAgent(state_dim, action_dim, buffer_capacity=buffer_capacity)
    env = MultiUserCSIEnv(
        data,
        mu_sigma=mu_sigma,
        base_checkpoint=base_checkpoint,
        allow_finetune=TRAIN_ALLOW_FINETUNE,
        use_online_normalizer=USE_ONLINE_NORMALIZER
    )
    print(f"\n--- AE Initialization Check ---")
    print(f"base_checkpoint type: {type(env.base_checkpoint)}")
    if isinstance(env.base_checkpoint, dict):
        for r, p in env.base_checkpoint.items():
            print(f"  ratio={r}: {p} exists={os.path.exists(p)}")
    print(f"AE latent dims at init: "
        f"{[env.aes[k].latent_dim for k in range(env.K)]}")
    
    logger = RewardLogger(name="rl_train")
    transitions = []
    total_steps = 0
    log_interval = 2
    
    # Visualizing Reward, scenario, action history
    reward_history = []  
    scenario_history = [] 
    action_history = []   
    train_epoch_avg_rewards = []
    greedy_eval_epoch_avg_rewards = []
    greedy_eval_epoch_histories = []
    best_greedy_eval_mean = -np.inf
    best_epoch_idx = -1
    epochs_without_improvement = 0
    debug_state_mean_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_qspread_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_loss_buf = deque(maxlen=max(1, RL_DEBUG_WINDOW * 4))
    debug_rate_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_nmse_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_bits_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_budget_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_env_reward_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_shaped_reward_buf = deque(maxlen=RL_DEBUG_WINDOW)
    debug_actions_buf = deque(maxlen=RL_DEBUG_WINDOW)
    
    N_TRAIN_EPOCHS = 10
    early_bonus_cutoff = max(1, (env.numSlots * N_TRAIN_EPOCHS) // 2)
    for epoch in range(N_TRAIN_EPOCHS):
        print(f"Epoch {epoch}:")
        start_idx = 0
        state = env.reset(start_idx=start_idx)
        episode_len = env.numSlots
        print(f"  start_idx={start_idx} episode_len={episode_len}")
        epoch_rewards = []
        for t in range(env.numSlots):
            
            actions = np.zeros(K, dtype=int)
            for k in range(K):
                state_k = build_joint_state_for_user(state, k)
                if RL_DEBUG:
                    with torch.no_grad():
                        s = torch.tensor(state_k, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        q_vals = agent.q_net(s).squeeze(0).cpu().numpy()
                    debug_qspread_buf.append(float(np.max(q_vals) - np.min(q_vals)))
                actions[k] = agent.select_action(state_k)
                
            next_state, rewards, done, info = env.step(actions)
            total_steps += 1
            reward_std = np.std(rewards)
            transition_done = done
            
            logger.log(info["reward_info"])
            reward_info = deepcopy(info["reward_info"])
            for k in range(K):
                state_k = build_joint_state_for_user(state, k)
                next_state_k = build_joint_state_for_user(next_state, k)
                agent.store(
                    state_k, actions[k], rewards[k], next_state_k, transition_done
                )
            for _ in range(GRAD_UPDATES_PER_STEP):
                loss_val = agent.train_step()
                if RL_DEBUG and loss_val is not None:
                    debug_loss_buf.append(float(loss_val))
            agent.decay_epsilon()

            reward_history.append(rewards)
            epoch_rewards.append(np.asarray(rewards, dtype=np.float32).copy())
            scenario_history.append([env.scenarioLabels[env.t-1, k] for k in range(env.K)])
            action_history.append(actions.copy())
            state_arr = np.asarray(state, dtype=np.float32)

            if RL_DEBUG:
                state_arr = np.asarray(state, dtype=np.float32)
                debug_state_mean_buf.append(np.mean(state_arr, axis=0))
                debug_actions_buf.append(actions.copy())
                debug_rate_buf.append(float(np.mean([reward_info[k]["rate"] for k in range(K)])))
                debug_nmse_buf.append(float(np.mean([reward_info[k]["nmse"] for k in range(K)])))
                debug_bits_buf.append(float(np.mean([reward_info[k]["bits"] for k in range(K)])))
                debug_budget_buf.append(float(np.mean([reward_info[k]["budget"] for k in range(K)])))
                debug_env_reward_buf.append(float(np.mean([info["reward_info"][k]["total_reward"] for k in range(K)])))
                debug_shaped_reward_buf.append(float(np.mean(rewards)))
                

                if total_steps % RL_DEBUG_INTERVAL == 0:
                    state_mean = np.mean(np.array(debug_state_mean_buf), axis=0) if debug_state_mean_buf else np.zeros(state_dim)
                    recent_actions = np.array(debug_actions_buf).reshape(-1) if debug_actions_buf else np.array([], dtype=int)
                    action_dist = {}
                    if recent_actions.size > 0:
                        vals, cts = np.unique(recent_actions, return_counts=True)
                        for a, c in zip(vals, cts):
                            action_dist[int(a)] = round(100.0 * float(c) / float(recent_actions.size), 2)

                    loss_mean = float(np.mean(debug_loss_buf)) if len(debug_loss_buf) > 0 else float("nan")
                    qspread_mean = float(np.mean(debug_qspread_buf)) if len(debug_qspread_buf) > 0 else float("nan")
                    print(
                        f"[RL-DEBUG] step={total_steps} eps={agent.epsilon:.4f} "
                        f"reward(shaped/env)={np.mean(debug_shaped_reward_buf):.4f}/{np.mean(debug_env_reward_buf):.4f} "
                        f"rate={np.mean(debug_rate_buf):.4f} nmse={np.mean(debug_nmse_buf):.4f} "
                        f"bits={np.mean(debug_bits_buf):.4f} budget={np.mean(debug_budget_buf):.4f} "
                        f"q_spread={qspread_mean:.4f} td_loss={loss_mean:.4f}"
                    )
                    print(
                        "[RL-DEBUG] joint_state_mean="
                        f" {np.round(state_mean, 4).tolist()}"
                    )
                    print(f"[RL-DEBUG] action_dist_recent={action_dist}")

            state = next_state
            if t % 50 == 0:
                print(f"Slot {t} | Avg Reward: {np.mean(rewards):.4f} | ε={agent.epsilon:.3f}")

            if done:
                break
        
        epoch_ckpt = os.path.join(SAVE_DIR, f"dqn_epoch_{epoch+1}.pt")
        torch.save(agent.q_net.state_dict(), epoch_ckpt)
        if len(epoch_rewards) == 0:
            train_epoch_avg = np.full(K, np.nan, dtype=np.float32)
        else:
            epoch_rewards_np = np.array(epoch_rewards, dtype=np.float32)
            train_epoch_avg = np.nanmean(epoch_rewards_np, axis=0)
        train_epoch_avg_rewards.append(train_epoch_avg)
        greedy_rewards, greedy_actions, greedy_scenarios = evaluate_greedy_policy(
            agent,
            data,
            mu_sigma,
            base_checkpoint
        )
        greedy_eval_avg = np.nanmean(greedy_rewards, axis=0)
        greedy_eval_epoch_avg_rewards.append(greedy_eval_avg)
        greedy_eval_epoch_histories.append({
            "epoch": epoch + 1,
            "reward_history": greedy_rewards,
            "action_history": greedy_actions,
            "scenario_history": greedy_scenarios
        })
        greedy_eval_mean = float(np.nanmean(greedy_eval_avg))
        if greedy_eval_mean > best_greedy_eval_mean + EARLY_STOPPING_MIN_DELTA:
            best_greedy_eval_mean = greedy_eval_mean
            best_epoch_idx = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "q_net_state_dict": agent.q_net.state_dict(),
                    "greedy_eval_avg": greedy_eval_avg,
                    "greedy_eval_mean": greedy_eval_mean
                },
                os.path.join(SAVE_DIR, "dqn_best.pt")
            )
        else:
            epochs_without_improvement += 1
        print(
            f"[Epoch {epoch + 1}] train_avg={np.round(train_epoch_avg, 4).tolist()} "
            f"greedy_eval_avg={np.round(greedy_eval_avg, 4).tolist()} "
            f"best_mean={best_greedy_eval_mean:.4f} patience={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
        )
        # if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        #     print(
        #         f"[Early Stopping] No greedy-eval improvement for {EARLY_STOPPING_PATIENCE} epochs. "
        #         f"Best epoch: {best_epoch_idx} with mean reward {best_greedy_eval_mean:.4f}"
        #     )
        #     break
    logger.save(SAVE_DIR)   
    torch.save(agent.q_net.state_dict(), os.path.join(SAVE_DIR, "dqn_final.pt"))
    # torch.save(env.aes[0].state_dict(), os.path.join(SAVE_DIR, "ae_final.pt"))
    ae_save_path = os.path.join(SAVE_DIR, "ae_final.pt")
    ae_ckpts = {
        k: {
            "state_dict": env.aes[k].state_dict(),
            "latent_dim": env.aes[k].latent_dim
        }
        for k in range(K)
    }
    torch.save(ae_ckpts, ae_save_path)

    # Saving the rewards of training just to avoid repeating training for camparison
    reward_RL_save_file = os.path.join(SAVE_DIR, "reward_history_training.pkl")
    action_RL_save_file = os.path.join(SAVE_DIR, "action_history_training.pkl")
    scenario_RL_save_file = os.path.join(SAVE_DIR, "scenario_history_training.pkl")
    train_epoch_avg_file = os.path.join(SAVE_DIR, "train_epoch_avg_rewards.pkl")
    greedy_eval_epoch_avg_file = os.path.join(SAVE_DIR, "greedy_eval_epoch_avg_rewards.pkl")
    greedy_eval_epoch_hist_file = os.path.join(SAVE_DIR, "greedy_eval_epoch_histories.pkl")
    with open(reward_RL_save_file, "wb") as f:
        pickle.dump(reward_history, f)
    with open(action_RL_save_file, "wb") as f:
        pickle.dump(action_history, f)
    with open(scenario_RL_save_file, "wb") as f:
        pickle.dump(scenario_history, f)
    with open(train_epoch_avg_file, "wb") as f:
        pickle.dump(np.array(train_epoch_avg_rewards), f)
    with open(greedy_eval_epoch_avg_file, "wb") as f:
        pickle.dump(np.array(greedy_eval_epoch_avg_rewards), f)
    with open(greedy_eval_epoch_hist_file, "wb") as f:
        pickle.dump(greedy_eval_epoch_histories, f)
    with open(os.path.join(SAVE_DIR, "best_epoch_summary.pkl"), "wb") as f:
        pickle.dump({
            "best_epoch": best_epoch_idx,
            "best_greedy_eval_mean": best_greedy_eval_mean,
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta": EARLY_STOPPING_MIN_DELTA
        }, f)
    
    plot_epoch_reward_summary(
        train_epoch_avg_rewards,
        greedy_eval_epoch_avg_rewards,
        os.path.join(SAVE_DIR, "RL_Training_Convergence.png"),
        best_epoch=best_epoch_idx,
        best_greedy_eval_mean=best_greedy_eval_mean
    )

    print('Training finished. Models saved.')


# ----------------- Evaluation ----------------
def evaluate_agent(data, final_ae_path, final_dqn_path, mu_sigma, paper_models):
    K = data['CSI'].shape[1]
    Nsub = data['CSI'].shape[4]
    NumBSant = data['CSI'].shape[2]
    in_dim = 2 * Nsub * NumBSant

    # === Load RL agent ===
    state_dim = STATE_DIM_PER_USER * K
    action_dim_per_user = TOTAL_ACTIONS
    agent = DQNAgent(state_dim, action_dim_per_user)
    
    dqn_ckpt = torch.load(final_dqn_path, map_location=DEVICE)
    if isinstance(dqn_ckpt, dict) and "q_net_state_dict" in dqn_ckpt:
        agent.q_net.load_state_dict(dqn_ckpt["q_net_state_dict"])
    else:
        agent.q_net.load_state_dict(dqn_ckpt)
    agent.q_net.eval()
    agent.epsilon = 0.0

    # RL INFERENCE LATENCY BENCHMARK  
    rl_policy_us  = benchmark_dqn_inference(
        q_net=agent.q_net,
        state_dim=state_dim,
        device=DEVICE
    )
    CSI_UPDATE_MS = 1.0  # conservative
    overhead_pct = (rl_policy_us  / 1000) / CSI_UPDATE_MS * 100
    print("\n--- RL Inference Benchmark ---")
    print(f"Average DQN inference latency: {rl_policy_us :.2f} µs")
    print(f"Overhead vs 1 ms CSI update: {overhead_pct:.3f}%")
    # Save the time
    with open(os.path.join(SAVE_DIR, "rl_policy_latency.txt"), "w") as f:
        f.write(f"Latency (us): {rl_policy_us :.3f}\n")
        f.write(f"Overhead (% of 1ms): {overhead_pct:.4f}\n")
    env = MultiUserCSIEnv(
        data,
        mu_sigma=mu_sigma,
        base_checkpoint=pretrain_paths,
        allow_finetune=False,
        use_online_normalizer=EVAL_USE_ONLINE_NORMALIZER
    )
    # Full RL LATENCY BENCHMARK  
    def _benchmark_rl_full_local(env_obj, agent_obj, num_runs=1000):
        state_local = env_obj.reset()
        start_local = time.perf_counter()
        for _ in range(num_runs):
            actions_local = np.zeros(env_obj.K, dtype=int)
            for kk in range(env_obj.K):
                state_local_k = build_joint_state_for_user(state_local, kk)
                actions_local[kk] = agent_obj.select_greedy_action(state_local_k)
            state_local, _, done_local, _ = env_obj.step(actions_local)
            if done_local:
                state_local = env_obj.reset()
        end_local = time.perf_counter()
        return (end_local - start_local) / num_runs * 1e6

    rl_full_us = _benchmark_rl_full_local(env, agent)
    
    overhead_pct = (rl_full_us  / 1000) / CSI_UPDATE_MS * 100
    print("\n--- RL Inference Benchmark ---")
    print(f"Full RL inference latency: {rl_full_us :.2f} µs")
    print(f"Overhead vs 1 ms CSI update: {overhead_pct:.3f}%")
    # Save the time
    with open(os.path.join(SAVE_DIR, "rl_full_latency.txt"), "w") as f:
        f.write(f"Latency (us): {rl_full_us :.3f}\n")
        f.write(f"Overhead (% of 1ms): {overhead_pct:.4f}\n")

    # Multi-user batch inference
    batch_state = torch.randn(K, state_dim, device=DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(5000):
            _ = agent.q_net(batch_state)
    end = time.perf_counter()

    batch_latency_us = (end - start) / 5000 * 1e6
    print(f"Batch inference latency for K={K}: {batch_latency_us:.2f} µs")    
    # ATH INFERENCE LATENCY BENCHMARK
    # Single-step average latency (per slot)
    ath_latency_us = benchmark_ath_batch_inference(env)
    ath_latency_us = ath_latency_us / K # To have it for one user
    overhead_pct = (ath_latency_us / 1000) / CSI_UPDATE_MS * 100

    print("\n--- ATH Batch Inference Benchmark ---")
    print(f"Average batch latency (all K users): {ath_latency_us:.2f} µs")
    print(f"Overhead vs 1 ms CSI update: {overhead_pct:.3f}%")
    with open(os.path.join(SAVE_DIR, "ath_latency.txt"), "w") as f:
        f.write(f"Batch Latency (us): {ath_latency_us:.3f}\n")
        f.write(f"Overhead (% of 1ms): {overhead_pct:.4f}\n")
    # DNN-baseline INFERENCE LATENCY BENCHMARK
    
    dnn_latency_us = benchmark_dnn_inference(models=paper_models, compression_list=COMPRESSION_RATIOS, state_dim=state_dim, mu=mu_sigma[0],
        sigma=mu_sigma[1], device=DEVICE)
    
    overhead_pct = (dnn_latency_us / 1000) / CSI_UPDATE_MS * 100

    print("\n--- LAD Inference Benchmark ---")
    print(f"Average batch latency (all K users): {dnn_latency_us:.2f} µs")
    print(f"Overhead vs 1 ms CSI update: {overhead_pct:.3f}%")
    with open(os.path.join(SAVE_DIR, "dnn_latency_us.txt"), "w") as f:
        f.write(f"Batch Latency (us): {dnn_latency_us:.3f}\n")
        f.write(f"Overhead (% of 1ms): {overhead_pct:.4f}\n")
    # fixed-CR baseline INFERENCE LATENCY BENCHMARK
    fixed_latency_us = benchmark_fixed_ratio_inference(
        env,
        fixed_ratio=1/8
    )
    fixed_latency_us = fixed_latency_us/ K # To have it for one user
    overhead_pct = (fixed_latency_us / 1000) / CSI_UPDATE_MS * 100
    
    print("\n--- Fixed Ratio Env Benchmark ---")
    print(f"Average batch latency (all K users): {fixed_latency_us:.2f} µs")
    print(f"Overhead vs 1 ms CSI update: {overhead_pct:.3f}%")
    with open(os.path.join(SAVE_DIR, "fixed_latency_us.txt"), "w") as f:
        f.write(f"Batch Latency (us): {fixed_latency_us:.3f}\n")
        f.write(f"Overhead (% of 1ms): {overhead_pct:.4f}\n")
    # === Environment ===
    env = MultiUserCSIEnv(
        data,
        mu_sigma=mu_sigma,
        base_checkpoint=pretrain_paths,
        allow_finetune=False,
        use_online_normalizer=EVAL_USE_ONLINE_NORMALIZER
    )
    state = env.reset()
    reward_history = []      # shape: [T, K]
    action_history = []      # shape: [T, K]
    scenario_history = []    # shape: [T]
    logger = RewardLogger(name="rl_eval")
    while True:
        actions = np.zeros(K, dtype=int)
        for k in range(K):
            state_k = build_joint_state_for_user(state, k)
            actions[k] = agent.select_greedy_action(state_k)

        next_state, rewards, done, info = env.step(actions)
        logger.log(info["reward_info"])
        reward_history.append(rewards)
        action_history.append(actions.copy())
        scenario_history.append([env.scenarioLabels[env.t-1, k] for k in range(env.K)])

        state = next_state
       
        if done:
            break

    reward_RL_save_file = os.path.join(SAVE_DIR, "reward_history_test.pkl")
    action_RL_save_file = os.path.join(SAVE_DIR, "action_history_test.pkl")
    scenario_RL_save_file = os.path.join(SAVE_DIR, "scenario_history_test.pkl")

    with open(reward_RL_save_file, "wb") as f:
        pickle.dump(reward_history, f)
    with open(action_RL_save_file, "wb") as f:
        pickle.dump(action_history, f)
    with open(scenario_RL_save_file, "wb") as f:
        pickle.dump(scenario_history, f)

    logger.save(SAVE_DIR)
    plot_rl_performance(
        reward_history,
        scenario_history,
        action_history
    )

# ------------------------ ATH Evaluation ----------------------------------
def evaluate_ath_baseline(data, mu_sigma, base_checkpoint):
    K = data['CSI'].shape[1]
    env = MultiUserCSIEnv(
        data,
        mu_sigma=mu_sigma,
        base_checkpoint=base_checkpoint,
        allow_finetune=False,
        use_online_normalizer=EVAL_USE_ONLINE_NORMALIZER
    )
    controllers = [ATHController() for _ in range(K)]
    for c in controllers:
        c.reset()
    ath_current_actions = np.zeros(K, dtype=int)

    state = env.reset()

    reward_history = []
    action_history = []
    scenario_history = []
    logger = RewardLogger(name="ath_baseline")
    while True:
        actions = np.zeros(K, dtype=int)

        for k in range(K):
            
            sinr_db = 10 * np.log10(env.sinr_db[k] + 1e-12)
            nmse_k  = env.last_nmse[k]
            # ATH uses its historical convention: 0 = hold, 1.. = select CR.
            # Convert to this env's direct convention: 0..3 = select CR.
            ath_raw = controllers[k].select_action(sinr_db, nmse_k)
            if ath_raw > 0:
                ath_current_actions[k] = int(np.clip(ath_raw - 1, 0, N_CR_ACTIONS - 1))
            actions[k] = ath_current_actions[k]

        next_state, rewards, done, info = env.step(actions)
        logger.log(info["reward_info"])
        reward_history.append(rewards)
        action_history.append(actions.copy())
        scenario_history.append([env.scenarioLabels[env.t-1, k] for k in range(K)])

        state = next_state
        if done:
            break
    logger.save(SAVE_DIR)
    return np.array(reward_history), np.array(action_history), np.array(scenario_history)

# ------------------------ Fixed CR Evaluation -----------------------------
def evaluate_fixed_ratio_baseline(data, mu_sigma, base_checkpoint, fixed_ratio=1/8):

    print(f"\n--- Evaluating Fixed Compression Baseline (CR={fixed_ratio}) ---")

    K = data['CSI'].shape[1]
    env = MultiUserCSIEnv(
        data,
        mu_sigma=mu_sigma,
        base_checkpoint=base_checkpoint,
        allow_finetune=False,
        use_online_normalizer=EVAL_USE_ONLINE_NORMALIZER
    )

    # Find index of the fixed ratio in compression list
    if fixed_ratio not in COMPRESSION_RATIOS:
        raise ValueError("Fixed ratio must be one of COMPRESSION_RATIOS")

    cr_idx   = COMPRESSION_RATIOS.index(fixed_ratio)
    joint_action = cr_idx

    state = env.reset()

    reward_history = []
    action_history = []
    scenario_history = []
    logger = RewardLogger(name=f"fixed_ratio_{fixed_ratio:.2f}")
    while True:

        # Same action for all users
        actions = np.ones(K, dtype=int) * joint_action 

        next_state, rewards, done, info = env.step(actions)
        logger.log(info["reward_info"])
        reward_history.append(rewards)
        action_history.append(actions.copy())
        scenario_history.append([env.scenarioLabels[env.t-1, k] for k in range(K)])

        state = next_state

        if done:
            break
    logger.save(SAVE_DIR)
    return (
        np.array(reward_history),
        np.array(action_history),
        np.array(scenario_history),
    )
def plot_all_rewards(rl_rewards, ath_rewards, paper_rewards, fixed_rewards_4, 
                     fixed_rewards_8, fixed_rewards_16, fixed_rewards_32, RL_action_history,
        RL_scenario_history):
    plt.figure(figsize=(16, 12))
    sns.set_style("whitegrid")

    # Calculate means
    rl_mean      = np.mean(rl_rewards,      axis=1)
    ath_mean     = np.mean(ath_rewards,     axis=1)
    paper_mean   = np.mean(paper_rewards,   axis=1)
    fixed_4_mean  = np.mean(fixed_rewards_4,  axis=1)
    fixed_8_mean  = np.mean(fixed_rewards_8,  axis=1)
    fixed_16_mean = np.mean(fixed_rewards_16, axis=1)
    fixed_32_mean = np.mean(fixed_rewards_32, axis=1)
    # Smoothing
    window = 50
    rl_smooth      = pd.Series(rl_mean).rolling(window=window).mean()
    ath_smooth     = pd.Series(ath_mean).rolling(window=window).mean()
    paper_smooth   = pd.Series(paper_mean).rolling(window=window).mean()
    fixed_4_smooth  = pd.Series(fixed_4_mean).rolling(window=window).mean()
    fixed_8_smooth  = pd.Series(fixed_8_mean).rolling(window=window).mean()
    fixed_16_smooth = pd.Series(fixed_16_mean).rolling(window=window).mean()
    fixed_32_smooth = pd.Series(fixed_32_mean).rolling(window=window).mean()
    x = np.arange(len(rl_mean))
    
    # Plot 1: Reward Evolution — spans both top columns
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(x, rl_smooth, label='RL (Proposed)', color='blue', linewidth=2)
    plt.plot(x, ath_smooth, label='ATH (Rule-Based)', color='green', linestyle='--', linewidth=2)
    plt.plot(x, paper_smooth, label='Adaptive DNN (Paper)', color='red', linestyle='-.', linewidth=2)
    plt.plot(x, fixed_4_smooth, label='Fixed 4-CR Baseline', color='purple', linestyle=':')
    plt.plot(x, fixed_8_smooth, label='Fixed 8-CR Baseline', color='black', linestyle=':')
    plt.plot(x, fixed_16_smooth, label='Fixed 16-CR Baseline', color='orange', linestyle=':')
    plt.plot(x, fixed_32_smooth, label='Fixed 32-CR Baseline', color='#33ffe7', linestyle=':')
    plt.ylabel("Average Reward")
    plt.title("Comparative Reward Evolution")
    plt.legend()
    
    # Plot 2: RL action strategy heatmap — decode direct CR actions
    raw_actions = np.array(RL_action_history).flatten()
    scenarios   = np.array(RL_scenario_history).flatten()
    scen_map    = {1: 'LoS', 2: 'NLoS', 3: 'Blockage'}

    cr_labels = {0: 'CR=1/4', 1: 'CR=1/8', 2: 'CR=1/16', 3: 'CR=1/32'}
    cr_actions = [cr_labels[int(a)] for a in raw_actions]

    df = pd.DataFrame({
        'CR_Action': cr_actions,
        'Scenario':  [scen_map.get(int(s), str(int(s))) for s in scenarios]
    })

    # Heatmap 1: CR selection frequency per scenario
    cr_counts = pd.crosstab(df['Scenario'], df['CR_Action'], normalize='index')
    # Reorder columns logically
    col_order = [c for c in ['CR=1/4', 'CR=1/8', 'CR=1/16', 'CR=1/32'] if c in cr_counts.columns]
    cr_counts = cr_counts[col_order]

    plt.subplot(2, 2, 3)
    sns.heatmap(cr_counts, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("CR Selection Frequency per Scenario")
    plt.xlabel("CR Action")
    plt.ylabel("Scenario")

    plt.subplot(2, 2, 4)
    plt.axis("off")
    plt.text(
        0.5, 0.5,
        "RL action space:\ndirectly choose one\npretrained AE ratio",
        ha="center", va="center", fontsize=12
    )

    plt.tight_layout()
    plt.savefig("Final_Comparison_with_RL_Strategy.png", dpi=300)
    plt.show()


def _slot_mean_rewards(reward_array):
    return np.mean(np.asarray(reward_array), axis=1)


def _find_scenario_change_points(scenario_history):
    scenario_arr = np.asarray(scenario_history)
    if scenario_arr.ndim == 1:
        scenario_arr = scenario_arr[:, None]

    change_points = []
    for t in range(1, len(scenario_arr)):
        if np.any(scenario_arr[t] != scenario_arr[t - 1]):
            change_points.append(t)
    return change_points


def _transition_metrics_for_method(
    reward_series,
    change_points,
    window,
    pre_window=50,
    recovery_tol_frac=0.05
):
    aligned_segments = []
    reward_drops = []
    recovery_steps = []

    for cp in change_points:
        if cp < pre_window or (cp + window) > len(reward_series):
            continue

        pre_segment = reward_series[cp - pre_window:cp]
        post_segment = reward_series[cp:cp + window]
        if len(pre_segment) < pre_window or len(post_segment) < window:
            continue

        baseline = float(np.mean(pre_segment))
        min_post = float(np.min(post_segment))
        reward_drops.append(max(0.0, baseline - min_post))

        recovery_tol = recovery_tol_frac * max(abs(baseline), 1.0)
        recovery_threshold = baseline - recovery_tol
        recovered_idx = np.where(post_segment >= recovery_threshold)[0]
        recovery_steps.append(
            int(recovered_idx[0]) if len(recovered_idx) > 0 else window
        )

        aligned_segments.append(np.concatenate([pre_segment, post_segment]))

    if not aligned_segments:
        return {
            "aligned_mean": None,
            "aligned_std": None,
            "reward_drop_mean": np.nan,
            "reward_drop_std": np.nan,
            "recovery_steps_mean": np.nan,
            "recovery_steps_std": np.nan,
            "num_events": 0
        }

    aligned_arr = np.vstack(aligned_segments)
    return {
        "aligned_mean": np.mean(aligned_arr, axis=0),
        "aligned_std": np.std(aligned_arr, axis=0),
        "reward_drop_mean": float(np.mean(reward_drops)),
        "reward_drop_std": float(np.std(reward_drops)),
        "recovery_steps_mean": float(np.mean(recovery_steps)),
        "recovery_steps_std": float(np.std(recovery_steps)),
        "num_events": len(aligned_segments)
    }


def plot_transition_window_analysis(
    reward_dict,
    scenario_history,
    window_sizes=(50, 100, 200),
    pre_window=50,
    save_prefix="Transition_Window_Analysis"
    ):
    method_styles = {
        "RL (Proposed)": dict(color="blue", linestyle="-", linewidth=2.2),
        "ATH (Rule-Based)": dict(color="green", linestyle="--", linewidth=2.0),
        "Adaptive DNN (Paper)": dict(color="red", linestyle="-.", linewidth=2.0),
        "Fixed 4-CR Baseline": dict(color="purple", linestyle=":", linewidth=1.8),
        "Fixed 8-CR Baseline": dict(color="black", linestyle=":", linewidth=1.8),
        "Fixed 16-CR Baseline": dict(color="orange", linestyle=":", linewidth=1.8),
        "Fixed 32-CR Baseline": dict(color="#33ffe7", linestyle=":", linewidth=1.8),
    }

    change_points = _find_scenario_change_points(scenario_history)
    if not change_points:
        print("[Transition Analysis] No scenario change points found. Skipping transition-window plots.")
        return {}

    metrics_by_window = {}
    fig, axes = plt.subplots(
        len(window_sizes), 3,
        figsize=(18, 5 * len(window_sizes)),
        squeeze=False
    )
    sns.set_style("whitegrid")

    for row_idx, window in enumerate(window_sizes):
        metrics_by_window[window] = {}

        for method_name, rewards in reward_dict.items():
            metrics = _transition_metrics_for_method(
                reward_series=_slot_mean_rewards(rewards),
                change_points=change_points,
                window=window,
                pre_window=pre_window
            )
            metrics_by_window[window][method_name] = metrics

        ax_curve = axes[row_idx, 0]
        time_axis = np.arange(-pre_window, window)
        for method_name, metrics in metrics_by_window[window].items():
            if metrics["aligned_mean"] is None:
                continue
            style = method_styles.get(method_name, {})
            ax_curve.plot(time_axis, metrics["aligned_mean"], label=method_name, **style)
            ax_curve.fill_between(
                time_axis,
                metrics["aligned_mean"] - metrics["aligned_std"],
                metrics["aligned_mean"] + metrics["aligned_std"],
                color=style.get("color", "gray"),
                alpha=0.08
            )
        ax_curve.axvline(0, color="gray", linestyle="--", linewidth=1.2)
        ax_curve.set_title(f"Aligned Reward Around Change (Post Window = {window})")
        ax_curve.set_xlabel("Steps Relative to Scenario Change")
        ax_curve.set_ylabel("Average Reward")
        if row_idx == 0:
            ax_curve.legend(fontsize=9, ncol=2)

        method_names = list(metrics_by_window[window].keys())
        drop_vals = [metrics_by_window[window][m]["reward_drop_mean"] for m in method_names]
        rec_vals = [metrics_by_window[window][m]["recovery_steps_mean"] for m in method_names]
        colors = [method_styles.get(m, {}).get("color", "gray") for m in method_names]

        ax_drop = axes[row_idx, 1]
        ax_drop.bar(method_names, drop_vals, color=colors, alpha=0.85)
        ax_drop.set_title(f"Reward Drop in First {window} Steps")
        ax_drop.set_ylabel("Baseline - Post-Change Minimum")
        ax_drop.tick_params(axis="x", rotation=35)

        ax_rec = axes[row_idx, 2]
        ax_rec.bar(method_names, rec_vals, color=colors, alpha=0.85)
        ax_rec.set_title(f"Recovery Speed in First {window} Steps")
        ax_rec.set_ylabel("Steps to Recover Near Pre-Change Reward")
        ax_rec.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300)
    plt.show()

    summary_rows = []
    for window, method_metrics in metrics_by_window.items():
        for method_name, metrics in method_metrics.items():
            summary_rows.append({
                "window": window,
                "method": method_name,
                "num_events": metrics["num_events"],
                "reward_drop_mean": metrics["reward_drop_mean"],
                "reward_drop_std": metrics["reward_drop_std"],
                "recovery_steps_mean": metrics["recovery_steps_mean"],
                "recovery_steps_std": metrics["recovery_steps_std"],
            })

    pd.DataFrame(summary_rows).to_csv(f"{save_prefix}_summary.csv", index=False)
    return metrics_by_window
def analyze_fixed_ratio_optima(
    fixed_reward_dict,
    scenario_history,
    save_prefix="Fixed_Ratio_Optima"
):
    scenario_arr = np.asarray(scenario_history)
    if scenario_arr.ndim == 1:
        scenario_arr = scenario_arr[:, None]

    reward_arrays = {
        name: np.asarray(rewards, dtype=np.float32)
        for name, rewards in fixed_reward_dict.items()
    }
    if not reward_arrays:
        print("[Fixed Ratio Analysis] No fixed-ratio rewards provided.")
        return {}

    first_shape = next(iter(reward_arrays.values())).shape
    if len(first_shape) != 2:
        print("[Fixed Ratio Analysis] Expected reward arrays with shape [T, K].")
        return {}

    T_common = min([scenario_arr.shape[0]] + [arr.shape[0] for arr in reward_arrays.values()])
    K_common = min([scenario_arr.shape[1]] + [arr.shape[1] for arr in reward_arrays.values()])
    scenario_arr = scenario_arr[:T_common, :K_common]
    reward_arrays = {name: arr[:T_common, :K_common] for name, arr in reward_arrays.items()}

    method_names = list(reward_arrays.keys())
    slot_means = {
        name: np.mean(arr, axis=1)
        for name, arr in reward_arrays.items()
    }

    summary = {}
    overall_means = {name: float(np.mean(arr)) for name, arr in reward_arrays.items()}
    overall_best = max(overall_means, key=overall_means.get)
    summary["overall"] = {
        "best_ratio": overall_best,
        "means": overall_means,
    }

    print("\n[Fixed Ratio Analysis] Overall mean reward by fixed ratio:")
    for name in method_names:
        print(f"  {name}: {overall_means[name]:.4f}")
    print(f"  -> overall best fixed ratio: {overall_best}")

    unique_scenarios = np.unique(scenario_arr[~np.isnan(scenario_arr)]).astype(int)
    scenario_rows = []
    user_rows = []

    if len(unique_scenarios) == 0:
        print("[Fixed Ratio Analysis] No valid scenario labels found.")
        return summary

    print("\n[Fixed Ratio Analysis] Best fixed ratio by scenario:")
    best_per_scenario = {}
    for scenario_id in unique_scenarios:
        slot_mask = np.any(scenario_arr == scenario_id, axis=1)
        num_slots = int(np.sum(slot_mask))
        if num_slots == 0:
            continue

        means = {
            name: float(np.mean(slot_means[name][slot_mask]))
            for name in method_names
        }
        best_name = max(means, key=means.get)
        best_per_scenario[int(scenario_id)] = best_name
        print(f"  scenario {scenario_id}: best={best_name} | slots={num_slots} | means={means}")

        scenario_rows.append({
            "level": "scenario_overall",
            "scenario": int(scenario_id),
            "user": -1,
            "best_ratio": best_name,
            "num_slots": num_slots,
            **means,
        })

        for user_idx in range(K_common):
            user_mask = scenario_arr[:, user_idx] == scenario_id
            user_slots = int(np.sum(user_mask))
            if user_slots == 0:
                continue

            user_means = {
                name: float(np.mean(reward_arrays[name][user_mask, user_idx]))
                for name in method_names
            }
            user_best = max(user_means, key=user_means.get)
            user_rows.append({
                "level": "scenario_user",
                "scenario": int(scenario_id),
                "user": int(user_idx),
                "best_ratio": user_best,
                "num_slots": user_slots,
                **user_means,
            })

    summary["best_per_scenario"] = best_per_scenario

    print("\n[Fixed Ratio Analysis] Best fixed ratio by scenario and user:")
    for row in user_rows:
        print(
            f"  scenario {row['scenario']} user {row['user']}: "
            f"best={row['best_ratio']} | slots={row['num_slots']}"
        )

    out_df = pd.DataFrame(scenario_rows + user_rows)
    out_path = f"{save_prefix}_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[Fixed Ratio Analysis] Saved summary to {out_path}")
    return summary
## ------------------------------- Main ------------------------------------
if __name__ == '__main__':
    # ... (Load Data) ...
    print('Loading dataset...')
    data = load_dataset(DATASET_PATH)
    if data is None:
        print("Dataset loading failed. Cannot proceed.")
        exit()
        
    T = data['CSI'].shape[0]
    K = data['CSI'].shape[1]
    test_slots = int(T * 0.3) # max(800, int(T * 0.1))
    train_data = {k: v[:T-test_slots, ...] if v is not None else None for k, v in data.items()}
    test_data = {k: v[T-test_slots:, ...] if v is not None else None for k, v in data.items()}
    print(f'Dataset loaded: K={K} Users, Train Slots={T-test_slots}, Test Slots={test_slots}')

    # Calculate Normalization Stats
    MU, SIGMA = calculate_normalization_stats(train_data)
    # ------------------------------------------------------------------
    train_mode = False  # Train/Test mode selection > True for train
    fixed_only = False  # When True in eval mode, run only fixed baselines + fixed-ratio diagnostic
    # ------------------------------------------------------------------
    if train_mode:
        pretrain_paths = pretrain_ae_per_ratio(
            train_data,
            COMPRESSION_RATIOS,
            save_dir=AE_CHECKPOINT_DIR,
            epochs=50,
            mu_sigma=(MU, SIGMA)
        )
    else:
        pretrain_paths = {
            cr: os.path.join(AE_CHECKPOINT_DIR, f"ae_cr_{int(1/cr)}.pt")
            for cr in COMPRESSION_RATIOS
        }
    start = time.time()

    if train_mode:
        used_data = train_data
    else:
        used_data = test_data
    
    paper_models = None
    paper_cr_classifier = None
    if not fixed_only:
        # LAD baseline: training models
        paper_models = train_paper_baseline_models(
            train_data,
            MU,
            SIGMA,
            COMPRESSION_RATIOS,
            epochs=10
        )
        paper_cr_classifier = train_cr_classifier(
            data=train_data,
            models=paper_models,
            mu=MU,
            sigma=SIGMA,
            compression_list=COMPRESSION_RATIOS,
            epochs=20,
            batch_size=256,
            margin=0.005
        )
    # --- RL agent training ---
    if train_mode: 
        train_agent_on_dataset(used_data, mu_sigma=(MU, SIGMA), base_checkpoint=pretrain_paths)
        print(f'Elapsed time (s): {time.time() - start:.2f}')
    elif not fixed_only:
        # --- RL agent Evaluation ---
        print('\nStarting Evaluation...')
        evaluate_agent(used_data, os.path.join(SAVE_DIR, "ae_final.pt"), os.path.join(SAVE_DIR, "dqn_best.pt"),
                        mu_sigma=(MU, SIGMA), paper_models=paper_models)
    
    
    # --- ATH Baseline Evaluation ---
    ath_rewards, ath_actions, ath_scenarios = None, None, None
    if not fixed_only:
        print("\nEvaluating ATH baseline...")
        ath_rewards, ath_actions, ath_scenarios = evaluate_ath_baseline(
            used_data,
            mu_sigma=(MU, SIGMA),
            base_checkpoint=pretrain_paths
        )
    # --- Fixed-CR Baseline Evaluation ---
    print("\nEvaluating Fixed Compression Baseline...")

    fixed_rewards_4, fixed_actions_4, fixed_scenarios_4 = evaluate_fixed_ratio_baseline(
        used_data,
        mu_sigma=(MU, SIGMA),
        base_checkpoint=pretrain_paths,
        fixed_ratio=1/4
    )
    fixed_rewards_8, fixed_actions_8, fixed_scenarios_8 = evaluate_fixed_ratio_baseline(
        used_data,
        mu_sigma=(MU, SIGMA),
        base_checkpoint=pretrain_paths,
        fixed_ratio=1/8
    )
    fixed_rewards_16, fixed_actions_16, fixed_scenarios_16 = evaluate_fixed_ratio_baseline(
        used_data,
        mu_sigma=(MU, SIGMA),
        base_checkpoint=pretrain_paths,
        fixed_ratio=1/16
    )
    fixed_rewards_32, fixed_actions_32, fixed_scenarios_32 = evaluate_fixed_ratio_baseline(
        used_data,
        mu_sigma=(MU, SIGMA),
        base_checkpoint=pretrain_paths,
        fixed_ratio=1/32
    )

    rl_rewards = None
    rl_action = None
    rl_scenario = None
    paper_rewards, paper_actions = None, None

    if not fixed_only:
        print("\nEvaluating RL agent...")
        reward_RL_save_file = os.path.join(SAVE_DIR, "reward_history_training.pkl")
        reward_RL_save_file_ = os.path.join(SAVE_DIR, "reward_history_test.pkl")
        action_RL_save_file = os.path.join(SAVE_DIR, "action_history_training.pkl")
        action_RL_save_file_ = os.path.join(SAVE_DIR, "action_history_test.pkl")
        scenario_RL_save_file = os.path.join(SAVE_DIR, "scenario_history_training.pkl")
        scenario_RL_save_file_ = os.path.join(SAVE_DIR, "scenario_history_test.pkl")
        reward_ATH_file = os.path.join(SAVE_DIR, "reward_ATH.pkl")

        if train_mode:
            with open(reward_RL_save_file, "rb") as f:
                RL_reward_history = pickle.load(f)
            with open(action_RL_save_file, "rb") as f:
                RL_action_history = pickle.load(f)
            with open(scenario_RL_save_file, "rb") as f:
                RL_scenario_history = pickle.load(f)
        else:
            with open(reward_RL_save_file_, "rb") as f:
                RL_reward_history_ = pickle.load(f)
            with open(action_RL_save_file_, "rb") as f:
                RL_action_history_ = pickle.load(f)
            with open(scenario_RL_save_file_, "rb") as f:
                RL_scenario_history_ = pickle.load(f)

        # --- PAPER BASELINE: Train & Evaluate ---
        paper_rewards, paper_actions = evaluate_paper_adaptive_baseline(
            used_data,
            paper_models,
            MU,
            SIGMA,
            COMPRESSION_RATIOS,
            cr_classifier=paper_cr_classifier
        )
    
    # --- Comparison Plotting ---
    if fixed_only:
        min_len = min(
            len(fixed_rewards_4),
            len(fixed_rewards_8),
            len(fixed_rewards_16),
            len(fixed_rewards_32),
            len(fixed_scenarios_4)
        )
        fixed_ratio_analysis = analyze_fixed_ratio_optima(
            fixed_reward_dict={
                "Fixed 4-CR Baseline": fixed_rewards_4[:min_len],
                "Fixed 8-CR Baseline": fixed_rewards_8[:min_len],
                "Fixed 16-CR Baseline": fixed_rewards_16[:min_len],
                "Fixed 32-CR Baseline": fixed_rewards_32[:min_len],
            },
            scenario_history=fixed_scenarios_4[:min_len],
            save_prefix="Fixed_Ratio_Optima"
        )
        print("Fixed-only diagnostic complete.")
        print(f"Elapsed time (s): {time.time() - start:.2f}")
        exit()

    if train_mode:
        rl_rewards = RL_reward_history
        rl_action = RL_action_history
        rl_scenario = RL_scenario_history
    else:
        rl_rewards = RL_reward_history_
        rl_action = RL_action_history_
        rl_scenario = RL_scenario_history_
    min_len = min(len(rl_rewards), len(ath_rewards), len(paper_rewards), len(fixed_rewards_4))
    fixed_ratio_analysis = analyze_fixed_ratio_optima(
        fixed_reward_dict={
            "Fixed 4-CR Baseline": fixed_rewards_4[:min_len],
            "Fixed 8-CR Baseline": fixed_rewards_8[:min_len],
            "Fixed 16-CR Baseline": fixed_rewards_16[:min_len],
            "Fixed 32-CR Baseline": fixed_rewards_32[:min_len],
        },
        scenario_history=rl_scenario[:min_len],
        save_prefix="Fixed_Ratio_Optima"
    )
    plot_all_rewards(
        rl_rewards[:min_len], 
        ath_rewards[:min_len], 
        paper_rewards[:min_len],
        fixed_rewards_4[:min_len],
        fixed_rewards_8[:min_len],
        fixed_rewards_16[:min_len],
        fixed_rewards_32[:min_len], 
        rl_action,
        rl_scenario
    )
    transition_metrics = plot_transition_window_analysis(
        reward_dict={
            "RL (Proposed)": rl_rewards[:min_len],
            "ATH (Rule-Based)": ath_rewards[:min_len],
            "Adaptive DNN (Paper)": paper_rewards[:min_len],
            "Fixed 4-CR Baseline": fixed_rewards_4[:min_len],
            "Fixed 8-CR Baseline": fixed_rewards_8[:min_len],
            "Fixed 16-CR Baseline": fixed_rewards_16[:min_len],
            "Fixed 32-CR Baseline": fixed_rewards_32[:min_len],
        },
        scenario_history=rl_scenario[:min_len],
        window_sizes=(50, 100, 200),
        pre_window=50,
        save_prefix="Transition_Window_Analysis"
    )
    # Save Paper results
    if train_mode:
        savemat(os.path.join(SAVE_DIR, "comparison_results_train.mat"), {
        "rl_rewards": rl_rewards[:min_len],
        "ath_rewards": ath_rewards[:min_len],
        "paper_rewards": paper_rewards[:min_len],
        "fixed_4": fixed_rewards_4[:min_len],
        "fixed_8": fixed_rewards_8[:min_len],
        "fixed_16": fixed_rewards_16[:min_len],
        "fixed_32": fixed_rewards_32[:min_len],
        "paper_actions": paper_actions[:min_len], 
        "rl_action": rl_action[:min_len],
        "rl_scenario": rl_scenario[:min_len],
        "ath_action": ath_actions[:min_len],
        "transition_event_count": len(_find_scenario_change_points(rl_scenario[:min_len]))
        })
    else:
        savemat(os.path.join(SAVE_DIR, "comparison_results_test.mat"), {
        "rl_rewards": rl_rewards[:min_len],
        "ath_rewards": ath_rewards[:min_len],
        "paper_rewards": paper_rewards[:min_len],
        "fixed_4": fixed_rewards_4[:min_len],
        "fixed_8": fixed_rewards_8[:min_len],
        "fixed_16": fixed_rewards_16[:min_len],
        "fixed_32": fixed_rewards_32[:min_len],
        "paper_actions": paper_actions[:min_len], 
        "rl_action": rl_action[:min_len],
        "rl_scenario": rl_scenario[:min_len],
        "ath_action": ath_actions[:min_len],
        "transition_event_count": len(_find_scenario_change_points(rl_scenario[:min_len]))
        })
    
    print("Comparison Complete.")
