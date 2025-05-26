# Using these two generated samples
# results_ddpm_2025-05-13_21-29.npy
# results_ddpm_2025-05-13_21-26.npy

# Using array to store filenames, ensuring each filename is passed as a separate parameter
# Using full file paths
FILES=(
  "results_ddpm_2025-05-13_21-29/samples.npy"
)

CUDA_VISIBLE_DEVICES=7 python inference_and_metrics.py "${FILES[@]}"

# Print log file path for viewing
echo "Check the latest log file for detailed results:"
ls -lt logs/inference_and_metrics_*.log | head -1




