# Bootstrapped MAE

This repository implements standard MAE and our bootstrapped MAE variant with feature-level reconstruction on CIFAR-10.

## 🚀 Getting Started

### 1. Create Conda Environment
```bash
conda create -n mae python=3.8 -y
conda activate mae
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Prepare Logging Directory
```bash
mkdir log
cd log
mkdir bmae_pretrain bmae_linprobe bmae_finetune mae_train mae_linprobe mae_finetune
cd ..
```

## 🏋️‍♂️ Training & Evaluation

### 4. Pretrain MAE
```bash
bash mae_train.sh
```
### 5. Evaluate MAE (Linear Probe & Finetune)
```bash
bash mae_eval_linear.sh
bash mae_eval_finetune.sh
```
### 6. Train Bootstrapped MAE
```bash
bash Bmae_train.sh
```
### 7. Evaluate Bootstrapped MAE (Linear Probe & Finetune)
```bash
bash Bmae_eval_linear.sh
bash Bmae_eval_finetune.sh
```
