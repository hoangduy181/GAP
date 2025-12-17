# Test Instructions

This document explains how to run a simple test to verify that training and evaluation work correctly after the fixes.

## Quick Test

Run the test script which will:
1. Train for 3 epochs (with debug mode using only 100 samples)
2. Save the model
3. Run evaluation on the saved model

### Option 1: Using Python script (Recommended)

```bash
# Basic test with default config
python3 test_train_eval.py

# Or specify a custom config
python3 test_train_eval.py --config config/nturgbd-cross-subject/kaggle_lst_joint.yaml

# Or specify device(s) for multi-GPU test
python3 test_train_eval.py --device "0 1"
```

### Option 2: Using bash script

```bash
bash test_train_eval.sh
```

### Option 3: Manual test

```bash
# Step 1: Training (3 epochs)
python3 main_multipart_ntu.py \
    --config config/test_minimal.yaml \
    --model model.ctrgcn.Model_lst_4part \
    --work-dir work_dir/test_run \
    --phase train \
    --num-epoch 3 \
    --save-interval 1 \
    --save-epoch 1 \
    --num-worker 2 \
    --device 0

# Step 2: Find the saved model
MODEL_FILE=$(find work_dir/test_run -name "runs-*.pt" | head -1)

# Step 3: Evaluation
python3 main_multipart_ntu.py \
    --config config/test_minimal.yaml \
    --model model.ctrgcn.Model_lst_4part \
    --work-dir work_dir/test_run \
    --phase test \
    --weights "$MODEL_FILE" \
    --device 0
```

## Test Config

The `config/test_minimal.yaml` is configured for quick testing:
- **Debug mode**: Uses only first 100 samples (fast)
- **Small batch size**: 8 (reduces memory usage)
- **3 epochs**: Quick validation
- **No augmentation**: Faster processing

**Note**: You may need to update the `data_path` in the config file to point to your actual dataset location.

## Multi-GPU Test

To test multi-GPU support (which was one of the fixes):

```bash
python3 test_train_eval.py --device "0 1"
```

Or manually:

```bash
python3 main_multipart_ntu.py \
    --config config/test_minimal.yaml \
    --model model.ctrgcn.Model_lst_4part \
    --work-dir work_dir/test_run \
    --phase train \
    --num-epoch 3 \
    --device 0 1
```

## What to Check

After running the test, verify:

1. ✅ **No import errors** - All modules load correctly
2. ✅ **Training runs** - Loss decreases, no CUDA errors
3. ✅ **Model saves** - Checkpoint files are created in work_dir
4. ✅ **Evaluation runs** - Accuracy is computed
5. ✅ **Multi-GPU works** - If using multiple devices, no "replica" errors

## Expected Output

You should see:
- Training progress with loss values
- Evaluation results with Top1/Top5 accuracy
- Model files saved in the work directory
- No CUDA memory errors or misaligned address errors

## Troubleshooting

### Data path not found
Update `data_path` in the config file to point to your dataset.

### CUDA out of memory
- Reduce `batch_size` in the config
- Use debug mode (already enabled in test_minimal.yaml)
- Use fewer GPUs

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements_updated.txt
```

