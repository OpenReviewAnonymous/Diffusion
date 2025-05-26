#!/bin/bash

# Create log directory
mkdir -p logs

echo "Start running inference evaluation..."
echo "=========================="

# # # Define model path variables
# # DDPM_MODEL="best_model_ddpm_2025-05-15_00-58-29.pth"
# # ADAPTIVE_DDPM_MODEL="best_model_adaptive_ddpm_2025-05-15_01-46.pth"
# # DIFFWAVE_MODEL="best_model_diffwave_2025-05-15_01:02.pth"
# # ADAPTIVE_DIFFWAVE_MODEL="best_model_adaptive_diffwave_2025-05-15_00-58.pth"

# # Define model path variables
# # DDPM_MODEL="best_model_ddpm_2025-05-15_05-04-29.pth"
# # ADAPTIVE_DDPM_MODEL="best_model_adaptive_ddpm_2025-05-15_05-07:35.pth"
# # DIFFWAVE_MODEL="best_model_diffwave_2025-05-15_01:02.pth"
# ADAPTIVE_DIFFWAVE_MODEL1="best_model_adaptive_diffwave_2025-05-21_08-00:10.pth"
# ADAPTIVE_DIFFWAVE_MODEL2="best_model_adaptive_diffwave_2025-05-21_08-06:15.pth"
# ADAPTIVE_DIFFWAVE_MODEL3="best_model_adaptive_diffwave_2025-05-21_08-12:21.pth"
# ADAPTIVE_DIFFWAVE_MODEL4="best_model_adaptive_diffwave_2025-05-21_08-18:27.pth"
# ADAPTIVE_DIFFWAVE_MODEL5="best_model_adaptive_diffwave_2025-05-21_08-24:33.pth"
# ADAPTIVE_DIFFWAVE_MODEL6="best_model_adaptive_diffwave_2025-05-21_08-30:39.pth"
# ADAPTIVE_DIFFWAVE_MODEL7="best_model_adaptive_diffwave_2025-05-21_08-36:45.pth"
# ADAPTIVE_DIFFWAVE_MODEL8="best_model_adaptive_diffwave_2025-05-21_08-42:51.pth"
# ADAPTIVE_DIFFWAVE_MODEL9="best_model_adaptive_diffwave_2025-05-21_08-48:57.pth"
# ADAPTIVE_DIFFWAVE_MODEL10="best_model_adaptive_diffwave_2025-05-21_08-55:03.pth"

# # # 1. Run standard DDPM evaluation
# # echo "Running standard DDPM evaluation..."
# # python univariate/evaluate_ddpm.py --model_path $DDPM_MODEL
# # echo "Standard DDPM evaluation completed"
# # echo "=========================="

# # # 2. Run adaptive DDPM evaluation
# # echo "Running adaptive DDPM evaluation..."
# # python univariate/evaluate_adaptive_ddpm.py --model_path $ADAPTIVE_DDPM_MODEL
# # echo "Adaptive DDPM evaluation completed"
# # echo "=========================="

# # # 3. Run DiffWave evaluation
# # echo "Running DiffWave evaluation..."
# # python univariate/evaluate_diffwave.py $DIFFWAVE_MODEL
# # echo "DiffWave evaluation completed"
# # echo "=========================="

# # 4. Run adaptive DiffWave evaluation
# echo "Running adaptive DiffWave evaluation..."
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL1
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL2
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL3
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL4
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL5
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL6
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL7
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL8
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL9
# python univariate/evaluate_adaptive_diffwave.py $ADAPTIVE_DIFFWAVE_MODEL10
# echo "Adaptive DiffWave evaluation completed"
# echo "=========================="

# 5. Run Adaptive DDPM regularity test evaluation
echo "=========================="
echo "Running Adaptive DDPM regularity test evaluation..."

# Set Python path
PYTHON_PATH="/the_path_to_python/python"

# Scheme A models (using new 06:43 run results)
echo "Evaluating scheme A models..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_A_2025-05-23_06-43:24_A_seed_1001.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_A_2025-05-23_06-43:26_A_seed_1002.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_A_2025-05-23_06-43:28_A_seed_1003.pth

# Scheme B models (using new 06:43 run results)
echo "Evaluating scheme B models..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_B_2025-05-23_06-43:30_B_seed_1001.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_B_2025-05-23_06-43:31_B_seed_1002.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_B_2025-05-23_06-43:34_B_seed_1003.pth

# Scheme C models (using new 06:43 run results)
echo "Evaluating scheme C models..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_C_2025-05-23_06-43:36_C_seed_1001.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_C_2025-05-23_06-43:38_C_seed_1002.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_C_2025-05-23_06-43:40_C_seed_1003.pth

# Scheme D models (newly added scheme D)
echo "Evaluating scheme D models..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_D_2025-05-23_06-43:42_D_seed_1001.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_D_2025-05-23_06-43:44_D_seed_1002.pth
CUDA_VISIBLE_DEVICES=2 $PYTHON_PATH univariate/evaluate_adaptive_ddpm_regularity_test.py --model_path best_model_adaptive_ddpm_D_2025-05-23_06-43:46_D_seed_1003.pth

echo "Adaptive DDPM regularity test evaluation completed"
echo "=========================="

echo "All inference evaluations completed"
echo "Results and log files are saved in the logs directory"
