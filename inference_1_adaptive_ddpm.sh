#!/bin/bash

# Create log directory
mkdir -p logs

echo "Starting inference evaluation..."
echo "=========================="

# Define model path variables
# ADAPTIVE_DDPM_MODEL1="best_model_adaptive_ddpm_2025-05-15_04-06:37.pth"
# ADAPTIVE_DDPM_MODEL2="best_model_adaptive_ddpm_2025-05-15_04-06:56.pth"
# ADAPTIVE_DDPM_MODEL3="best_model_adaptive_ddpm_2025-05-15_04-07:08.pth"
# ADAPTIVE_DDPM_MODEL4="best_model_adaptive_ddpm_2025-05-15_04-07:31.pth"

# ADAPTIVE_DDPM_MODEL1="best_model_adaptive_ddpm_2025-05-15_04-20:26.pth"
# ADAPTIVE_DDPM_MODEL2="best_model_adaptive_ddpm_2025-05-15_04-21:34.pth"
# ADAPTIVE_DDPM_MODEL3="best_model_adaptive_ddpm_2025-05-15_04-22:09.pth"

# ADAPTIVE_DDPM_MODEL1="best_model_adaptive_ddpm_2025-05-15_04-23:04.pth"

# ADAPTIVE_DDPM_MODEL1="best_model_adaptive_ddpm_2025-05-15_04-39:04.pth"

# ADAPTIVE_DDPM_MODEL1="best_model_adaptive_ddpm_2025-05-15_04-43:29.pth"

# ADAPTIVE_DDPM_MODEL1="best_model_adaptive_ddpm_2025-05-15_04-48:34.pth"
# ADAPTIVE_DDPM_MODEL2="best_model_adaptive_ddpm_2025-05-15_04-50:02.pth"
# ADAPTIVE_DDPM_MODEL3="best_model_adaptive_ddpm_2025-05-15_04-50:23.pth"

# 2. Run adaptive DDPM evaluation
echo "Running adaptive DDPM evaluation..."
python univariate/evaluate_adaptive_ddpm.py --model_path $ADAPTIVE_DDPM_MODEL1
echo "Adaptive DDPM evaluation completed"
echo "=========================="

# 2. Run adaptive DDPM evaluation
echo "Running adaptive DDPM evaluation..."
python univariate/evaluate_adaptive_ddpm.py --model_path $ADAPTIVE_DDPM_MODEL2
echo "Adaptive DDPM evaluation completed"
echo "=========================="

# # 2. Run adaptive DDPM evaluation
# echo "Running adaptive DDPM evaluation..."
# python univariate/evaluate_adaptive_ddpm.py --model_path $ADAPTIVE_DDPM_MODEL3
# echo "Adaptive DDPM evaluation completed"
# echo "=========================="
