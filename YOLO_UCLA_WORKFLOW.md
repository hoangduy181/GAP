# YOLO Pose UCLA Dataset Workflow

This document describes the complete workflow for processing UCLA multiview action videos with YOLO pose detection.

## Directory Structure

Your videos are organized as:
```
data/multiview_action_videos/
├── a01/          # Action class 1
│   ├── v01_s01_e00.avi
│   ├── v01_s01_e01.avi
│   └── ...
├── a02/          # Action class 2
│   └── ...
└── ...
```

## Step 1: Create Train/Val Split (70:30)

First, create the train/val split based on your video directory structure:

```bash
python create_ucla_split.py \
    --video-dir data/multiview_action_videos \
    --output-dir data/ucla_splits \
    --train-ratio 0.7 \
    --seed 42
```

This will create:
- `data/ucla_splits/train_split.json` - List of training videos with labels
- `data/ucla_splits/val_split.json` - List of validation videos with labels
- `data/ucla_splits/split_summary.json` - Summary statistics

The split ensures:
- 70% of videos go to training, 30% to validation
- Balanced distribution across all action classes
- Reproducible splits (using seed)

## Step 2: Preprocess Training Videos

Extract YOLO keypoints from training videos:

```bash
python main_multipart_yolo_ucla.py \
    --video-dir data/multiview_action_videos \
    --output-dir data/ucla_yolo/train \
    --split train \
    --split-file data/ucla_splits/train_split.json \
    --yolo-model-path yolo11n-pose.pt \
    --yolo-conf-threshold 0.25
```

## Step 3: Preprocess Validation Videos

Extract YOLO keypoints from validation videos:

```bash
python main_multipart_yolo_ucla.py \
    --video-dir data/multiview_action_videos \
    --output-dir data/ucla_yolo/val \
    --split val \
    --split-file data/ucla_splits/val_split.json \
    --yolo-model-path yolo11n-pose.pt \
    --yolo-conf-threshold 0.25
```

## Step 4: Update Config File

Update `config/ucla/yolo_pose.yaml` to point to the preprocessed data:

```yaml
train_feeder_args:
  data_path: data/ucla_yolo/train  # Directory with preprocessed JSON files
  label_path: train

test_feeder_args:
  data_path: data/ucla_yolo/val  # Directory with preprocessed JSON files
  label_path: val
```

## Step 5: Train Model

Train using the preprocessed data:

```bash
python main_multipart_ucla.py \
    --config config/ucla/yolo_pose.yaml \
    --work-dir work_dir/ucla/yolo_pose \
    --device 0
```

## Output Format

Each preprocessed video is saved as a JSON file with:
- `skeletons`: List of frames, each frame contains 17 joints [x, y, confidence]
- `metadata`: Video information (fps, dimensions, label, etc.)

The format is compatible with the existing UCLA feeder structure.

## Notes

- The preprocessing step only needs to be run once
- Videos are automatically skipped if already processed (unless `--overwrite` is used)
- The split ensures balanced distribution across classes
- Labels are extracted from directory names (a01 -> label 0, a02 -> label 1, etc.)
