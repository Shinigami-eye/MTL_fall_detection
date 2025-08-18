#!/bin/bash
# Run all experiments for UMAFall MTL pipeline

echo "=========================================="
echo "Running UMAFall MTL Experiments"
echo "=========================================="

# Prepare data if not already done
if [ ! -d "data/processed" ]; then
    echo "Preparing data..."
    python scripts/prepare_data.py --config configs/dataset.yaml
fi

# Minimal Viable Set (MVS) experiments
echo -e "\n>>> Running MVS experiments..."

# 1. Single-task baseline: Fall only
echo -e "\n1. ST-Fall baseline..."
python scripts/train.py --config configs/experiments/mvs_baseline_st_fall.yaml

# 2. Single-task baseline: ADL only  
echo -e "\n2. ST-ADL baseline..."
python scripts/train.py --config configs/experiments/mvs_baseline_st_adl.yaml

# 3. MTL with equal weights
echo -e "\n3. MTL with equal weights (0.5/0.5)..."
python scripts/train.py --config configs/experiments/mvs_mtl_shared_05_05.yaml

# 4. MTL with uncertainty weighting
echo -e "\n4. MTL with uncertainty weighting..."
python scripts/train.py --config configs/experiments/mvs_mtl_uncertainty.yaml

# 5. MTL with GradNorm
echo -e "\n5. MTL with GradNorm..."
python scripts/train.py --config configs/experiments/mvs_mtl_gradnorm.yaml

# Full experiment grid
echo -e "\n>>> Running full experiment grid..."

# Task weight variations
echo -e "\n6. MTL with weights (0.3/0.7)..."
python scripts/train.py --config configs/experiments/exp_mtl_weights_03_07.yaml

echo -e "\n7. MTL with weights (0.7/0.3)..."
python scripts/train.py --config configs/experiments/exp_mtl_weights_07_03.yaml

# Focal loss experiments
echo -e "\n8. Focal loss (gamma=1)..."
python scripts/train.py --config configs/experiments/exp_focal_loss_gamma1.yaml

echo -e "\n9. Focal loss (gamma=2)..."
python scripts/train.py --config configs/experiments/exp_focal_loss_gamma2.yaml

# Robustness experiments
echo -e "\n10. LOSO evaluation..."
python scripts/train.py --config configs/experiments/exp_loso.yaml

echo -e "\n11. Window size 1.5s..."
python scripts/train.py --config configs/experiments/exp_window_1_5s.yaml

echo -e "\n12. Window size 4s..."
python scripts/train.py --config configs/experiments/exp_window_4s.yaml

echo -e "\n13. With magnitude channel..."
python scripts/train.py --config configs/experiments/exp_magnitude_on.yaml

# Evaluate all models
echo -e "\n>>> Evaluating all models..."
for checkpoint in checkpoints/*/best_model.pt; do
    exp_name=$(basename $(dirname $checkpoint))
    echo -e "\nEvaluating $exp_name..."
    python scripts/evaluate.py --checkpoint $checkpoint --output reports/${exp_name}
done

# Run error analysis on best model
echo -e "\n>>> Running error analysis..."
python scripts/analyze_errors.py \
    --checkpoint checkpoints/mvs_mtl_gradnorm/best_model.pt \
    --baseline checkpoints/mvs_baseline_st_fall/best_model.pt

# Verify runs
echo -e "\n>>> Verifying experiment runs..."
python scripts/verify_run.py

echo -e "\n=========================================="
echo "All experiments complete!"
echo "Results saved in reports/"
echo "=========================================="
