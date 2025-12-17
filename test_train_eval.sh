#!/bin/bash
# Simple test script to verify training and evaluation work

set -e  # Exit on error

echo "=========================================="
echo "Testing Training and Evaluation"
echo "=========================================="

# Check if CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Set environment variable for CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Test config path
CONFIG="config/test_minimal.yaml"
WORK_DIR="work_dir/test_run"

# Clean up previous test run if exists
if [ -d "$WORK_DIR" ]; then
    echo "Cleaning up previous test run..."
    rm -rf "$WORK_DIR"
fi

echo ""
echo "Step 1: Running training (3 epochs with debug mode)..."
echo "=========================================="

python3 main_multipart_ntu.py \
    --config "$CONFIG" \
    --model model.ctrgcn.Model_lst_4part \
    --work-dir "$WORK_DIR" \
    --phase train \
    --num-epoch 3 \
    --save-interval 1 \
    --save-epoch 1 \
    --num-worker 2 \
    --device 0 \
    --print-log True

echo ""
echo "Step 2: Checking if model was saved..."
echo "=========================================="

# Find the best model
MODEL_FILE=$(find "$WORK_DIR" -name "runs-*.pt" | head -1)

if [ -z "$MODEL_FILE" ]; then
    echo "ERROR: No model file found!"
    exit 1
fi

echo "Found model: $MODEL_FILE"

echo ""
echo "Step 3: Running evaluation..."
echo "=========================================="

python3 main_multipart_ntu.py \
    --config "$CONFIG" \
    --model model.ctrgcn.Model_lst_4part \
    --work-dir "$WORK_DIR" \
    --phase test \
    --weights "$MODEL_FILE" \
    --device 0 \
    --print-log True

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="

