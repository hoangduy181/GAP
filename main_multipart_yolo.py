#!/usr/bin/env python
"""
Main script for YOLO-based pose detection and action recognition on NTU RGB+D 60 dataset.

This script supports three modes:
1. preprocess: Extract skeleton keypoints from videos using YOLO pose detection
2. train: Train action recognition model on preprocessed skeleton data
3. eval: Evaluate trained model on videos or preprocessed skeleton JSON files

Usage:
    # Preprocess videos
    python main_multipart_yolo.py --mode preprocess --config config/ntu60_yolo.yaml
    
    # Train model
    python main_multipart_yolo.py --mode train --config config/ntu60_yolo.yaml
    
    # Evaluate on videos
    python main_multipart_yolo.py --mode eval --config config/ntu60_yolo.yaml --eval-mode video --video path/to/video.avi
    
    # Evaluate on skeleton JSON files
    python main_multipart_yolo.py --mode eval --config config/ntu60_yolo.yaml --eval-mode skeleton --skeleton-dir data/ntu60_json/val
"""

from __future__ import print_function

import argparse
import inspect
import os
import glob
import json
import re
import random
import shutil
import sys
import time
import pickle
from collections import OrderedDict
import traceback
import yaml
from tqdm import tqdm
import numpy as np

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from torchlight import DictAction
from tools import *
from Text_Prompt import *
from KLLoss import KLLoss
from model.baseline import TextCLIP
from sklearn.metrics import confusion_matrix
import clip

from yolo_pose.yolo_pose_detector import YOLOPoseDetector
from yolo_pose.video_utils import get_video_metadata

# NTU60 text prompts
classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part()
text_list = text_prompt_openai_random()

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def init_seed(seed):
    """Initialize random seed for reproducibility."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def import_class(import_str):
    """Dynamically import a class from a module string."""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    """Convert string to boolean."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_ntu_filename(filename: str) -> dict:
    """
    Parse NTU60 video filename to extract components.
    
    Format: S{setup}C{camera}P{performer}R{repetition}A{action}_rgb.avi
    Example: S001C001P001R001A001_rgb.avi
    
    Args:
        filename: Video filename (with or without extension)
    
    Returns:
        Dictionary with parsed components or None if parsing fails
    """
    name_without_ext = os.path.splitext(filename)[0]
    name_without_ext = name_without_ext.replace('_rgb', '')
    
    match = re.match(r'S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)', name_without_ext)
    if match:
        return {
            'setup': int(match.group(1)),
            'camera': int(match.group(2)),
            'performer': int(match.group(3)),
            'repetition': int(match.group(4)),
            'action': int(match.group(5)),
            'label_action': int(match.group(5)) - 1,  # 0-indexed
            'label_subject': int(match.group(3)) - 1   # 0-indexed
        }
    return None


def preprocess_videos(config_path: str, debug: bool = False):
    """
    Preprocess mode: Extract skeleton keypoints from videos using YOLO pose detection.
    
    This function:
    1. Reads video paths from config file (data.json format)
    2. Extracts skeleton keypoints using YOLO pose detection
    3. Saves extracted keypoints as JSON files
    4. Updates data.json with json_path references
    
    Args:
        config_path: Path to config file
        debug: If True, process only first 100 samples for faster testing
    """
    print("="*60)
    print("PREPROCESS MODE: Extracting skeleton keypoints from videos")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    processing_config = config_data.get('processing', {})
    if not processing_config:
        raise ValueError("No 'processing' section found in config file")
    
    # Get paths from config
    data_config = config_data.get('data_config', {})
    data_json_path = data_config.get('data_json')
    if not data_json_path:
        raise ValueError("'data_config.data_json' not specified in config")
    
    video_dir = processing_config.get('video_dir')
    output_dir_base = processing_config.get('output_dir', 'data/ntu60_json')
    
    # YOLO parameters
    yolo_model_path = processing_config.get('yolo_model_path', 'yolo11n-pose.pt')
    yolo_conf_threshold = processing_config.get('yolo_conf_threshold', 0.25)
    yolo_device = processing_config.get('yolo_device')
    yolo_tracking_strategy = processing_config.get('yolo_tracking_strategy', 'largest_bbox')
    overwrite = processing_config.get('overwrite', False)
    
    # Load data.json to get video list
    if not os.path.exists(data_json_path):
        raise FileNotFoundError(f"data.json not found: {data_json_path}")
    
    with open(data_json_path, 'r') as f:
        data_json = json.load(f)
    
    # Initialize YOLO detector
    print(f"Loading YOLO model from {yolo_model_path}...")
    detector = YOLOPoseDetector(
        model_path=yolo_model_path,
        conf_threshold=yolo_conf_threshold,
        device=yolo_device,
        tracking_strategy=yolo_tracking_strategy
    )
    print("YOLO model loaded successfully.")
    
    # Process each split (train and val)
    for split_info in data_json.get('splits', []):
        split_name = split_info.get('split')
        items = split_info.get('items', [])
        
        if not items:
            print(f"No items found for split: {split_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split: {len(items)} videos")
        print(f"{'='*60}")
        
        # Debug mode: limit to first 100 samples
        if debug:
            items = items[:100]
            print(f"DEBUG MODE: Processing only first {len(items)} samples")
        
        # Create output directory for this split
        output_dir = os.path.join(output_dir_base, split_name)
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Process each video
        for item in tqdm(items, desc=f"Processing {split_name}"):
            video_path = item.get('full_path') or os.path.join(video_dir, item['video_path'])
            video_name = item['video_name']
            label_action = item['label_action']
            
            # Output JSON file path
            output_file = os.path.join(output_dir, f"{video_name}.json")
            
            # Skip if already exists and not overwriting
            if os.path.exists(output_file) and not overwrite:
                skipped_count += 1
                continue
            
            # Process video
            try:
                # Extract keypoints with YOLO
                return_frames = debug  # Return frames for visualization in debug mode
                if return_frames:
                    keypoints_list, metadata, frames = detector.detect_video(video_path, return_frames=True)
                else:
                    keypoints_list, metadata = detector.detect_video(video_path, return_frames=False)
                    frames = None
                
                # Convert to JSON format
                skeletons_array = np.array(keypoints_list)  # (T, 17, 3)
                skeletons_list = skeletons_array.tolist()
                
                # Save as JSON
                output_data = {
                    "skeletons": skeletons_list,
                    "metadata": {
                        "num_frames": len(keypoints_list),
                        "num_joints": 17,
                        "video_path": video_path,
                        "video_name": video_name,
                        "label_action": label_action,
                        "label_subject": item.get('label_subject', 0),
                        "fps": metadata.get('fps', 0),
                        "frame_count": metadata.get('frame_count', len(keypoints_list)),
                        "width": metadata.get('width', 0),
                        "height": metadata.get('height', 0)
                    }
                }
                
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                # Debug mode: Save visualization
                if debug and frames is not None:
                    try:
                        from yolo_pose.visualization import visualize_keypoints_on_video, save_keypoint_frames
                        
                        vis_dir = os.path.join(output_dir, 'visualizations')
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        vis_video_path = os.path.join(vis_dir, f"{video_name}_skeleton.mp4")
                        visualize_keypoints_on_video(
                            video_path=video_path,
                            keypoints_list=keypoints_list,
                            output_path=vis_video_path,
                            fps=metadata.get('fps', 30.0),
                            conf_threshold=yolo_conf_threshold
                        )
                        
                        vis_frames_dir = os.path.join(vis_dir, f"{video_name}_frames")
                        save_keypoint_frames(
                            video_path=video_path,
                            keypoints_list=keypoints_list,
                            output_dir=vis_frames_dir,
                            sample_interval=10,
                            conf_threshold=yolo_conf_threshold
                        )
                    except Exception as e:
                        print(f"  Warning: Failed to create visualization: {e}")
                
                # Update item with json_path
                item['json_path'] = output_file
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                error_count += 1
                continue
        
        print(f"\n{split_name} split processing complete:")
        print(f"  Processed: {processed_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Errors: {error_count}")
    
    # Save updated data.json with json_path references
    with open(data_json_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    print(f"\nUpdated data.json with json_path references: {data_json_path}")
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


def check_skeleton_extracted(config_path: str) -> bool:
    """
    Check if skeleton data has been extracted for all videos in config.
    
    Args:
        config_path: Path to config file
    
    Returns:
        True if all skeletons are extracted, False otherwise
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    data_config = config_data.get('data_config', {})
    data_json_path = data_config.get('data_json')
    
    if not data_json_path or not os.path.exists(data_json_path):
        return False
    
    with open(data_json_path, 'r') as f:
        data_json = json.load(f)
    
    missing_count = 0
    for split_info in data_json.get('splits', []):
        items = split_info.get('items', [])
        for item in items:
            json_path = item.get('json_path')
            if not json_path or not os.path.exists(json_path):
                missing_count += 1
    
    if missing_count > 0:
        print(f"Warning: {missing_count} skeleton JSON files are missing")
        return False
    
    return True


def run_training(config_path: str, debug: bool = False):
    """
    Training mode: Train action recognition model on preprocessed skeleton data.
    
    This function:
    1. Checks if skeleton data has been extracted
    2. Loads data using the configured feeder
    3. Trains the model with validation during training
    4. Saves model checkpoints
    
    Args:
        config_path: Path to config file
        debug: If True, use debug mode (reduces dataset to 100 samples)
    """
    print("="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Check if skeletons are extracted
    if not check_skeleton_extracted(config_path):
        print("\nERROR: Skeleton data not found. Please run preprocessing first:")
        print(f"  python main_multipart_yolo.py --mode preprocess --config {config_path}")
        return
    
    print("✓ Skeleton data found, proceeding with training...")
    
    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    training_config = config_data.get('training', {})
    if not training_config:
        raise ValueError("No 'training' section found in config file")
    
    # Create temporary flattened config for argparse
    import tempfile
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    
    flattened_config = {}
    for key, value in training_config.items():
        flattened_config[key] = value
    
    # Enable debug mode if requested
    if debug:
        if 'train_feeder_args' not in flattened_config:
            flattened_config['train_feeder_args'] = {}
        if 'test_feeder_args' not in flattened_config:
            flattened_config['test_feeder_args'] = {}
        flattened_config['train_feeder_args']['debug'] = True
        flattened_config['test_feeder_args']['debug'] = True
        print("DEBUG MODE: Enabled debug mode in feeder (will use first 100 samples)")
    
    yaml.dump(flattened_config, temp_config, default_flow_style=False)
    temp_config.close()
    
    try:
        # Get parser
        parser = get_training_parser()
        
        # Load config
        with open(temp_config.name, 'r') as f:
            default_arg = yaml.safe_load(f)
        
        # Parse args
        p = parser.parse_args(['--config', temp_config.name])
        parser.set_defaults(**default_arg)
        arg = parser.parse_args(['--config', temp_config.name])
        
        # Ensure critical fields are set
        if not arg.model and 'model' in default_arg:
            arg.model = default_arg['model']
        if not arg.feeder and 'feeder' in default_arg:
            arg.feeder = default_arg['feeder']
        
        # Merge nested dicts
        if 'model_args' in default_arg and isinstance(default_arg['model_args'], dict):
            if not hasattr(arg, 'model_args') or not isinstance(arg.model_args, dict):
                arg.model_args = default_arg['model_args']
            else:
                arg.model_args.update(default_arg['model_args'])
        
        if 'train_feeder_args' in default_arg and isinstance(default_arg['train_feeder_args'], dict):
            if not hasattr(arg, 'train_feeder_args') or not isinstance(arg.train_feeder_args, dict):
                arg.train_feeder_args = default_arg['train_feeder_args']
            else:
                arg.train_feeder_args.update(default_arg['train_feeder_args'])
        
        if 'test_feeder_args' in default_arg and isinstance(default_arg['test_feeder_args'], dict):
            if not hasattr(arg, 'test_feeder_args') or not isinstance(arg.test_feeder_args, dict):
                arg.test_feeder_args = default_arg['test_feeder_args']
            else:
                arg.test_feeder_args.update(default_arg['test_feeder_args'])
        
        # Pass data.json path to feeder args
        data_config = config_data.get('data_config', {})
        data_json_path = data_config.get('data_json')
        if data_json_path and os.path.exists(data_json_path):
            if hasattr(arg, 'train_feeder_args') and isinstance(arg.train_feeder_args, dict):
                arg.train_feeder_args['data_json_path'] = data_json_path
            if hasattr(arg, 'test_feeder_args') and isinstance(arg.test_feeder_args, dict):
                arg.test_feeder_args['data_json_path'] = data_json_path
        
        # Initialize and run
        init_seed(getattr(arg, 'seed', 0))
        processor = Processor(arg)
        processor.start()
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_config.name):
            os.unlink(temp_config.name)


def eval_from_video(config_path: str, weights_path: str, video_path: str = None,
                    video_list_path: str = None, save_score: bool = False):
    """
    Evaluation mode: Evaluate model on video file(s) by extracting skeleton on-the-fly.
    
    Args:
        config_path: Path to config file
        weights_path: Path to trained model weights
        video_path: Path to single video file (optional)
        video_list_path: Path to text file with video paths (optional)
        save_score: Whether to save prediction scores
    """
    if not video_path and not video_list_path:
        raise ValueError("Either --video or --video-list must be provided")
    
    print("="*60)
    print("EVALUATION MODE: Evaluating from videos")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    training_config = config_data.get('training', {})
    processing_config = config_data.get('processing', {})
    
    # Get video paths
    video_paths = []
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_paths.append(video_path)
    if video_list_path:
        if not os.path.exists(video_list_path):
            raise FileNotFoundError(f"Video list file not found: {video_list_path}")
        with open(video_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if os.path.exists(line):
                        video_paths.append(line)
    
    if not video_paths:
        raise ValueError("No valid video files found")
    
    print(f"Found {len(video_paths)} video(s) to evaluate")
    
    # Load model
    model_class = training_config.get('model')
    model_args = training_config.get('model_args', {})
    
    device_list = training_config.get('device', [0])
    output_device = device_list[0] if isinstance(device_list, list) and len(device_list) > 0 else 0
    
    Model = import_class(model_class)
    model_args_for_model = {k: v for k, v in model_args.items() if k != 'head'}
    model = Model(**model_args_for_model)
    model = model.cuda(output_device) if torch.cuda.is_available() else model
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    weights = torch.load(weights_path, map_location=f'cuda:{output_device}' if torch.cuda.is_available() else 'cpu')
    if any(k.startswith('module.') for k in weights.keys()):
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    
    # Initialize YOLO detector
    yolo_model_path = processing_config.get('yolo_model_path', 'yolo11n-pose.pt')
    yolo_conf_threshold = processing_config.get('yolo_conf_threshold', 0.25)
    yolo_device = processing_config.get('yolo_device')
    yolo_tracking_strategy = processing_config.get('yolo_tracking_strategy', 'largest_bbox')
    
    detector = YOLOPoseDetector(
        model_path=yolo_model_path,
        conf_threshold=yolo_conf_threshold,
        device=yolo_device,
        tracking_strategy=yolo_tracking_strategy
    )
    
    # Get window_size from config
    test_feeder_args = training_config.get('test_feeder_args', {})
    window_size = test_feeder_args.get('window_size', 64)
    
    # Create feeder instance for conversion utilities
    # We'll use the conversion method directly
    def yolo_to_ctrgcn_format(keypoints_list):
        """Convert YOLO keypoints to CTR-GCN format."""
        if not keypoints_list:
            return np.zeros((2, 1, 17, 1), dtype=np.float32)
        keypoints_stack = np.stack(keypoints_list, axis=0)
        keypoints_xy = keypoints_stack[:, :, :2]
        keypoints_ctrgcn = keypoints_xy.transpose(2, 0, 1)
        return keypoints_ctrgcn[:, :, :, np.newaxis].astype(np.float32)
    
    # Process each video
    all_predictions = []
    all_scores = []
    
    for idx, vid_path in enumerate(tqdm(video_paths, desc="Processing videos")):
        print(f"\n[{idx+1}/{len(video_paths)}] Processing: {os.path.basename(vid_path)}")
        
        # Extract keypoints with YOLO
        keypoints_list, metadata = detector.detect_video(vid_path, return_frames=False)
        print(f"  Extracted {len(keypoints_list)} frames")
        
        if len(keypoints_list) == 0:
            print(f"  Warning: No keypoints extracted")
            all_predictions.append((vid_path, None, None))
            continue
        
        # Convert to CTR-GCN format
        def yolo_to_ctrgcn_format_local(keypoints_list):
            """Convert YOLO keypoints to CTR-GCN format."""
            if not keypoints_list:
                return np.zeros((2, 1, 17, 1), dtype=np.float32)
            keypoints_stack = np.stack(keypoints_list, axis=0)
            keypoints_xy = keypoints_stack[:, :, :2]
            keypoints_ctrgcn = keypoints_xy.transpose(2, 0, 1)
            return keypoints_ctrgcn[:, :, :, np.newaxis].astype(np.float32)
        
        keypoints_array = yolo_to_ctrgcn_format_local(keypoints_list)
        
        # Apply temporal cropping/resizing
        from feeders import tools
        valid_frame_num = np.sum(keypoints_array.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            print(f"  Warning: No valid frames")
            all_predictions.append((vid_path, None, None))
            continue
        
        keypoints_array = tools.valid_crop_resize(
            keypoints_array,
            valid_frame_num,
            [0.95],
            window_size
        )
        
        # Add batch dimension
        keypoints_tensor = torch.from_numpy(keypoints_array).float().unsqueeze(0)
        if torch.cuda.is_available():
            keypoints_tensor = keypoints_tensor.cuda(output_device)
        
        # Run inference
        with torch.no_grad():
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                output, _, _, _ = model(keypoints_tensor)
        
        # Get predictions
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        all_class_scores = probabilities[0].cpu().numpy()
        
        all_predictions.append((vid_path, predicted_class, confidence))
        all_scores.append(all_class_scores)
        
        print(f"  Predicted class: {predicted_class}, Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for vid_path, pred_class, conf in all_predictions:
        if pred_class is not None:
            print(f"{os.path.basename(vid_path)}: Class {pred_class} (Conf: {conf:.4f})")
        else:
            print(f"{os.path.basename(vid_path)}: FAILED")
    
    # Save scores if requested
    if save_score:
        output_file = weights_path.replace('.pt', '_eval_scores.txt')
        with open(output_file, 'w') as f:
            f.write("Video Path,Predicted Class,Confidence,All Class Scores\n")
            for (vid_path, pred_class, conf), scores in zip(all_predictions, all_scores):
                if pred_class is not None:
                    scores_str = ','.join([f"{s:.6f}" for s in scores])
                    f.write(f"{vid_path},{pred_class},{conf:.6f},{scores_str}\n")
        print(f"\nScores saved to: {output_file}")
    
    print("="*60)


def eval_from_skeleton(config_path: str, weights_path: str, skeleton_dir: str = None,
                      save_score: bool = False):
    """
    Evaluation mode: Evaluate model on preprocessed skeleton JSON files.
    
    Args:
        config_path: Path to config file
        weights_path: Path to trained model weights
        skeleton_dir: Directory containing skeleton JSON files (optional, uses config if not provided)
        save_score: Whether to save prediction scores
    """
    print("="*60)
    print("EVALUATION MODE: Evaluating from skeleton JSON files")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    training_config = config_data.get('training', {})
    
    # Get skeleton directory from config if not provided
    if not skeleton_dir:
        data_config = config_data.get('data_config', {})
        output_dir_base = config_data.get('processing', {}).get('output_dir', 'data/ntu60_json')
        skeleton_dir = os.path.join(output_dir_base, 'val')  # Default to val split
    
    if not os.path.exists(skeleton_dir):
        raise FileNotFoundError(f"Skeleton directory not found: {skeleton_dir}")
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(skeleton_dir, '*.json'))
    json_files = [f for f in json_files if not f.endswith('_data_dict.json')]
    
    if not json_files:
        raise ValueError(f"No skeleton JSON files found in {skeleton_dir}")
    
    print(f"Found {len(json_files)} skeleton JSON files")
    
    # Load model
    model_class = training_config.get('model')
    model_args = training_config.get('model_args', {})
    
    device_list = training_config.get('device', [0])
    output_device = device_list[0] if isinstance(device_list, list) and len(device_list) > 0 else 0
    
    Model = import_class(model_class)
    model_args_for_model = {k: v for k, v in model_args.items() if k != 'head'}
    model = Model(**model_args_for_model)
    model = model.cuda(output_device) if torch.cuda.is_available() else model
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    weights = torch.load(weights_path, map_location=f'cuda:{output_device}' if torch.cuda.is_available() else 'cpu')
    if any(k.startswith('module.') for k in weights.keys()):
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()
    
    # Get window_size from config
    test_feeder_args = training_config.get('test_feeder_args', {})
    window_size = test_feeder_args.get('window_size', 64)
    
    # Process each JSON file
    all_predictions = []
    all_scores = []
    all_labels = []
    
    from feeders import tools
    
    for json_file in tqdm(json_files, desc="Processing skeleton files"):
        try:
            with open(json_file, 'r') as f:
                skeleton_data = json.load(f)
            
            skeletons = skeleton_data.get('skeletons', [])
            metadata = skeleton_data.get('metadata', {})
            label_action = metadata.get('label_action', 0)
            
            if not skeletons:
                print(f"Warning: No skeletons in {json_file}")
                continue
            
            # Convert to numpy array: (T, 17, 3)
            keypoints_list = [np.array(frame) for frame in skeletons]
            
            # Convert to CTR-GCN format
            def yolo_to_ctrgcn_format(keypoints_list):
                """Convert YOLO keypoints to CTR-GCN format."""
                if not keypoints_list:
                    return np.zeros((2, 1, 17, 1), dtype=np.float32)
                keypoints_stack = np.stack(keypoints_list, axis=0)
                keypoints_xy = keypoints_stack[:, :, :2]
                keypoints_ctrgcn = keypoints_xy.transpose(2, 0, 1)
                return keypoints_ctrgcn[:, :, :, np.newaxis].astype(np.float32)
            
            keypoints_array = yolo_to_ctrgcn_format(keypoints_list)
            
            # Apply temporal cropping/resizing
            valid_frame_num = np.sum(keypoints_array.sum(0).sum(-1).sum(-1) != 0)
            if valid_frame_num == 0:
                continue
            
            keypoints_array = tools.valid_crop_resize(
                keypoints_array,
                valid_frame_num,
                [0.95],
                window_size
            )
            
            # Add batch dimension
            keypoints_tensor = torch.from_numpy(keypoints_array).float().unsqueeze(0)
            if torch.cuda.is_available():
                keypoints_tensor = keypoints_tensor.cuda(output_device)
            
            # Run inference
            with torch.no_grad():
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    output, _, _, _ = model(keypoints_tensor)
            
            # Get predictions
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            predicted_class = predicted_class.item()
            confidence = confidence.item()
            all_class_scores = probabilities[0].cpu().numpy()
            
            all_predictions.append((os.path.basename(json_file), predicted_class, confidence))
            all_scores.append(all_class_scores)
            all_labels.append(label_action)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Calculate accuracy
    correct = sum(1 for (_, pred, _), label in zip(all_predictions, all_labels) if pred == label)
    accuracy = correct / len(all_predictions) if all_predictions else 0.0
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples: {len(all_predictions)}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nSample predictions:")
    for (name, pred_class, conf), label in zip(all_predictions[:10], all_labels[:10]):
        status = "✓" if pred_class == label else "✗"
        print(f"  {status} {name}: Pred={pred_class}, True={label}, Conf={conf:.4f}")
    
    # Save scores if requested
    if save_score:
        output_file = weights_path.replace('.pt', '_eval_skeleton_scores.txt')
        with open(output_file, 'w') as f:
            f.write("File Name,Predicted Class,True Label,Confidence,Correct,All Class Scores\n")
            for (name, pred_class, conf), label, scores in zip(all_predictions, all_labels, all_scores):
                correct_flag = "1" if pred_class == label else "0"
                scores_str = ','.join([f"{s:.6f}" for s in scores])
                f.write(f"{name},{pred_class},{label},{conf:.6f},{correct_flag},{scores_str}\n")
        print(f"\nScores saved to: {output_file}")
    
    print("="*60)


def get_training_parser():
    """Get argument parser for training mode."""
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if true, the classification score will be stored')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-alpha', type=float, default=1.0)
    parser.add_argument('--te-lr-ratio', type=float, default=1)
    return parser


# Copy Processor class from main_multipart_yolo_ucla.py
# (This is a large class, so we'll import the key methods)
# For now, let's create a simplified version that works with the new structure

class Processor():
    """ 
    Processor for Skeleton-based Action Recognition
    
    Handles model loading, data loading, training, and evaluation.
    """
    
    def __init__(self, arg):
        """Initialize Processor with configuration arguments."""
        self.arg = arg
        self.save_arg()
        
        if arg.phase == 'train':
            # Ensure debug key exists in train_feeder_args
            if 'debug' not in arg.train_feeder_args:
                arg.train_feeder_args['debug'] = False
            if not arg.train_feeder_args['debug']:
                tensorboard_dir = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(tensorboard_dir):
                    print('log_dir: ', tensorboard_dir, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(tensorboard_dir)
                        print('Dir removed: ', tensorboard_dir)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', tensorboard_dir)
                self.train_writer = SummaryWriter(os.path.join(tensorboard_dir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(tensorboard_dir, 'val'), 'val')
            else:
                tensorboard_dir = os.path.join(arg.work_dir, 'runs')
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(tensorboard_dir, 'test'), 'test')
        
        self.global_step = 0
        self.load_model()
        
        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        
        self.print_log(f'Processor initialized: {self.arg.work_dir}')
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        
        self.model = self.model.cuda(self.output_device)
        
        # Multi-GPU setup
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
        
        # Multi-GPU setup for text encoders
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                if 'head' not in self.arg.model_args:
                    raise KeyError("'head' key not found in model_args. Please check your config file.")
                if not self.arg.model_args['head']:
                    raise ValueError("model_args['head'] is empty. At least one CLIP head must be specified.")
                
                for name in self.arg.model_args['head']:
                    if name not in self.model_text_dict:
                        raise KeyError(f"Text encoder model '{name}' not found in model_text_dict. "
                                     f"Available models: {list(self.model_text_dict.keys())}")
                    
                    if not next(self.model_text_dict[name].parameters()).is_cuda:
                        self.model_text_dict[name] = self.model_text_dict[name].cuda(self.output_device)
                    
                    self.model_text_dict[name] = nn.DataParallel(
                        self.model_text_dict[name],
                        device_ids=self.arg.device,
                        output_device=self.output_device)
    
    def load_data(self):
        """Load data using the configured feeder."""
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.print_log(f'DataLoader initialized')
    
    def load_model(self):
        """Load model and text encoders (CLIP) for contrastive learning."""
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        if torch.cuda.is_available():
            self.output_device = output_device
        else:
            self.output_device = None
            print("WARNING: CUDA not available, using CPU")
        
        if self.arg.model is None:
            raise ValueError(f"Model is None. Please check your config file.")
        
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        
        # Filter out 'head' from model_args as it's used by Processor, not Model
        model_args_for_model = {k: v for k, v in self.arg.model_args.items() if k != 'head'}
        self.model = Model(**model_args_for_model)
        
        # Load CLIP text encoders
        self.model_text_dict = {}
        if 'head' not in self.arg.model_args:
            raise KeyError("'head' key not found in model_args. Please check your config file.")
        
        for name in self.arg.model_args['head']:
            model_text, _ = clip.load(name, device=device, jit=False)
            model_text.float()
            model_text.eval()
            self.model_text_dict[name] = model_text
        
        # Load weights if provided
        if self.arg.weights:
            self.print_log(f'Load weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            
            weights = OrderedDict([[k.split('module.')[-1], v.cuda(self.output_device)] for k, v in weights.items()])
            
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
    
    def load_optimizer(self):
        """Load optimizer for training."""
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                [{'params': self.model.parameters(),'lr': self.arg.base_lr},
                {'params': self.model_text_dict.parameters(), 'lr': self.arg.base_lr*self.arg.te_lr_ratio}],
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
        
        # Initialize loss functions
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss = KLLoss()
    
    def save_arg(self):
        """Save configuration arguments to work directory."""
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)
    
    def adjust_learning_rate(self, epoch):
        """Adjust learning rate based on epoch and schedule."""
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()
    
    def print_time(self):
        """Print current local time."""
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)
    
    def print_log(self, str, print_time=True):
        """Print log message with optional timestamp."""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)
    
    def record_time(self):
        """Record current time for timing measurements."""
        self.cur_time = time.time()
        return self.cur_time
    
    def split_time(self):
        """Calculate time elapsed since last record_time call."""
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    def train(self, epoch, save_model=False):
        """Train model for one epoch."""
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        
        loss_value = []
        loss_ce_value = []
        loss_te_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=100, desc=f'Epoch {epoch+1}', 
                      miniters=1, mininterval=0.1, 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output, feature_dict, logit_scale, part_feature_list = self.model(data)
                label_g = gen_label(label)
                label = label.long().cuda(self.output_device)
                loss_te_list = []
                for ind in range(num_text_aug):
                    if ind > 0:
                        text_id = np.ones(len(label),dtype=np.int8) * ind
                        text_tensors = [text_dict[j][i,:].contiguous() for i,j in zip(label,text_id)]
                        texts = torch.stack(text_tensors).cuda(self.output_device).contiguous()
                    else:
                        texts = list()
                        for i in range(len(label)):
                            text_len = len(text_list[label[i]])
                            text_id = np.random.randint(text_len,size=1)
                            text_item = text_list[label[i]][text_id.item()]
                            texts.append(text_item)
                        texts = [t.contiguous() if isinstance(t, torch.Tensor) else t for t in texts]
                        texts = torch.cat(texts).cuda(self.output_device).contiguous()
                    
                    text_embedding = self.model_text_dict[self.arg.model_args['head'][0]](texts).float()
                    
                    if ind == 0:
                        logits_per_image, logits_per_text = create_logits(feature_dict[self.arg.model_args['head'][0]],text_embedding,logit_scale[:,0].mean())
                        label_g_contiguous = np.ascontiguousarray(label_g)
                        ground_truth = torch.from_numpy(label_g_contiguous).to(dtype=feature_dict[self.arg.model_args['head'][0]].dtype, device=self.output_device).contiguous()
                    else:
                        logits_per_image, logits_per_text = create_logits(part_feature_list[ind-1],text_embedding,logit_scale[:,ind].mean())
                        label_g_contiguous = np.ascontiguousarray(label_g)
                        ground_truth = torch.from_numpy(label_g_contiguous).to(dtype=part_feature_list[ind-1].dtype, device=self.output_device).contiguous()
                    
                    loss_imgs = self.loss(logits_per_image,ground_truth)
                    loss_texts = self.loss(logits_per_text,ground_truth)
                    loss_te_list.append((loss_imgs + loss_texts) / 2)
                
                loss_ce = self.loss_ce(output, label)
                loss_te_avg = sum(loss_te_list) / len(loss_te_list)
                loss = loss_ce + self.arg.loss_alpha * loss_te_avg
            
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            if self.global_step % 100 == 0:
                torch.cuda.empty_cache()
            
            loss_value.append(loss.data.item())
            loss_ce_value.append(loss_ce.data.item())
            loss_te_value.append(loss_te_avg.data.item())
            timer['model'] += self.split_time()
            
            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_ce', loss_ce.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_te', loss_te_avg.data.item(), self.global_step)
            
            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    'Batch[{}/{}]: Loss: {:.4f} (CE: {:.4f}, TE: {:.4f}), Acc: {:.2f}%'.format(
                        batch_idx + 1, len(loader), 
                        loss.data.item(), loss_ce.data.item(), loss_te_avg.data.item(),
                        acc.data.item() * 100))
            
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
        
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        mean_loss = np.mean(loss_value)
        mean_loss_ce = np.mean(loss_ce_value)
        mean_loss_te = np.mean(loss_te_value)
        mean_acc = np.mean(acc_value) * 100
        
        self.print_log(
            '\tMean training loss: {:.4f} (CE: {:.4f}, TE: {:.4f}).  Mean training acc: {:.2f}%.'.format(
                mean_loss, mean_loss_ce, mean_loss_te, mean_acc))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))
        
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            model_filename = os.path.join(self.arg.work_dir, 'runs-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
            torch.save(weights, model_filename)
    
    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        """Evaluate model on validation/test set."""
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=100, desc=f'Evaluating {ln}',
                          miniters=1, mininterval=0.1,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    b, _, _, _, _ = data.size()
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    with torch.cuda.amp.autocast():
                        output, _, _, _ = self.model(data)
                    loss = self.loss_ce(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1
                
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1
            
            print('Accuracy: ', accuracy, ' model: ', self.arg.work_dir)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)
            
            mean_loss = np.mean(loss_value)
            self.print_log('\tMean {} loss of {} batches: {:.4f} (classification only).'.format(
                ln, len(self.data_loader[ln]), mean_loss))
            for k in self.arg.show_topk:
                accuracy_k = self.data_loader[ln].dataset.top_k(score, k)
                self.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy_k))
                if self.arg.phase == 'train':
                    self.val_writer.add_scalar('acc_top{}'.format(k), accuracy_k, self.global_step)
        
        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()
    
    def start(self):
        """Start training or evaluation process."""
        self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        
        def count_parameters(model):
            """Count trainable parameters in model."""
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if self.arg.phase == 'train':
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch
                
                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            
            # Find best model
            if self.best_acc_epoch > 0:
                weights_files = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))
                if not weights_files:
                    all_weights = glob.glob(os.path.join(self.arg.work_dir, 'runs-*.pt'))
                    if all_weights:
                        weights_files = [max(all_weights, key=os.path.getmtime)]
                        self.print_log(f'Using most recent model: {os.path.basename(weights_files[0])}')
                    else:
                        raise FileNotFoundError(f'No model weights found in {self.arg.work_dir}.')
                weights_path = weights_files[0]
            else:
                all_weights = glob.glob(os.path.join(self.arg.work_dir, 'runs-*.pt'))
                if not all_weights:
                    raise FileNotFoundError(f'No model weights found in {self.arg.work_dir}.')
                weights_path = max(all_weights, key=os.path.getmtime)
                self.print_log(f'Using most recent model: {os.path.basename(weights_path)}')
            
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)
            
            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True
            
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
        
        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')
            
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def get_main_parser():
    """Get argument parser for main script."""
    parser = argparse.ArgumentParser(
        description='YOLO-based pose detection and action recognition')
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['preprocess', 'train', 'eval'],
        required=True,
        help='Mode: preprocess (extract skeletons), train (train model), or eval (evaluate model)')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file')
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: reduce dataset to 100 samples for faster testing')
    
    # Evaluation mode arguments
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=['video', 'skeleton'],
        help='Evaluation mode: video (extract skeleton on-the-fly) or skeleton (use preprocessed JSON files)')
    
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to trained model weights (required for eval mode)')
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to single video file (for eval-mode video)')
    
    parser.add_argument(
        '--video-list',
        type=str,
        help='Path to text file with video paths (for eval-mode video)')
    
    parser.add_argument(
        '--skeleton-dir',
        type=str,
        help='Directory containing skeleton JSON files (for eval-mode skeleton)')
    
    parser.add_argument(
        '--save-score',
        action='store_true',
        help='Save prediction scores to file (for eval mode)')
    
    return parser


if __name__ == '__main__':
    parser = get_main_parser()
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        preprocess_videos(args.config, debug=args.debug)
    
    elif args.mode == 'train':
        run_training(args.config, debug=args.debug)
    
    elif args.mode == 'eval':
        if not args.weights:
            raise ValueError("--weights is required for eval mode")
        
        if args.eval_mode == 'video':
            eval_from_video(
                config_path=args.config,
                weights_path=args.weights,
                video_path=args.video,
                video_list_path=args.video_list,
                save_score=args.save_score
            )
        elif args.eval_mode == 'skeleton':
            eval_from_skeleton(
                config_path=args.config,
                weights_path=args.weights,
                skeleton_dir=args.skeleton_dir,
                save_score=args.save_score
            )
        else:
            raise ValueError("--eval-mode must be specified for eval mode (video or skeleton)")
