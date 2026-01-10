# YOLO UCLA Training Guide

## Overview

The training process has two main components:
1. **Preprocessing** (`main_multipart_yolo_ucla.py`) - Extracts YOLO keypoints from videos
2. **Training** (`main_multipart_ucla.py`) - Trains the model using preprocessed data

## Complete Workflow

### Step 1: Create Train/Val Split

```bash
python create_ucla_split.py \
    --video-dir data/multiview_action_videos \
    --output-dir data/ucla_splits \
    --train-ratio 0.7
```

This creates:
- `data/ucla_splits/train_split.json`
- `data/ucla_splits/val_split.json`
- `data/ucla_splits/split_summary.json`

### Step 2: Preprocess Videos (Using Config)

The preprocessing script now reads from the config file:

```bash
# Process training videos
python main_multipart_yolo_ucla.py --config config/ucla/yolo_pose.yaml --split train

# Process validation videos
python main_multipart_yolo_ucla.py --config config/ucla/yolo_pose.yaml --split val
```

The config file (`config/ucla/yolo_pose.yaml`) contains all preprocessing parameters:
- `preprocessing.video_dir` - Root directory with action class folders
- `preprocessing.output_dir` - Base directory for preprocessed JSON files
- `preprocessing.split_file_dir` - Directory containing split files
- `preprocessing.train_split_file` - Path to train split file
- `preprocessing.val_split_file` - Path to val split file
- `preprocessing.overwrite` - Whether to overwrite existing files
- YOLO parameters (model_path, conf_threshold, device, tracking_strategy)

### Step 3: Train Model

Training happens in `main_multipart_ucla.py`:

```bash
python main_multipart_ucla.py \
    --config config/ucla/yolo_pose.yaml \
    --work-dir work_dir/ucla/yolo_pose \
    --device 0
```

The training script:
- **Automatically preprocesses val videos** if they don't exist (checks `preprocessing.auto_preprocess_val` in config)
- Loads the feeder specified in config (`feeders.feeder_yolo_pose_ucla.Feeder`)
- Uses preprocessed JSON files from `data_path` in config
- Trains the model and evaluates on val set during training
- Trains the model using the specified parameters

**Note:** If val JSON files are missing, the training script will automatically run preprocessing for val videos before starting training. This ensures you always have val data for evaluation during training.

## Config File Structure

The `config/ucla/yolo_pose.yaml` file contains:

1. **Preprocessing section** - For `main_multipart_yolo_ucla.py`
2. **Feeder section** - Data loading parameters
3. **Model section** - Model architecture
4. **Training section** - Training hyperparameters

## Key Points

- **Preprocessing is separate from training** - Run once before training
- **Training loads preprocessed JSON files** - No video processing during training
- **Config file centralizes all parameters** - Easy to manage and modify
- **Command line args override config** - Flexible usage

## File Flow

```
Videos (multiview_action_videos/)
    ↓ [create_ucla_split.py]
Split files (ucla_splits/)
    ↓ [main_multipart_yolo_ucla.py]
Preprocessed JSON (ucla_yolo/)
    ↓ [main_multipart_ucla.py]
Trained Model (work_dir/)
```
