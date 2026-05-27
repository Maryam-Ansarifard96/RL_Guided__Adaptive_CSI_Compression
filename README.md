# Dynamic Compression-Ratio Control Using RL-Guided Autoencoder Switching for CSI Feedback in Multi-User MIMO Systems
This repository contains the official implementation of the paper:

**"Dynamic Compression-Ratio Control Using RL-Guided Autoencoder Switching for CSI Feedback in Multi-User MIMO Systems"**

The proposed framework dynamically adjusts CSI compression ratios using reinforcement learning-guided switching among multiple autoencoders in multi-user MIMO systems.


## Installation

```bash
git clone https://github.com/Maryam-Ansarifard96/RL_Guided__Adaptive_CSI_Compression.git
cd RL_Guided__Adaptive_CSI_Compression
pip install -r requirements.txt
```
GPU acceleration is supported through CUDA-enabled PyTorch.
## Dataset Generation

The dataset used in this work is generated using the MATLAB script:

```text
csiGeneration.m
```
Run the MATLAB script before training or evaluation to generate the CSI dataset required by the framework.

Example:

```matlab
run('csiGeneration.m')
```

The generated dataset will be saved as a `.mat` file and later loaded by the Python implementation.

---

## Training and Evaluation

The main implementation is executed through:

```text
main.py
```

Inside `main.py`, switch between training and testing modes using:

```python
train_mode = True   # Training mode
train_mode = False  # Evaluation / testing mode
```
