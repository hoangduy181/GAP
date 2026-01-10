#!/usr/bin/env python
"""
Preprocessing script to extract YOLO pose keypoints from UCLA video files
and save them in a format compatible with the training pipeline.

Workflow:
    1. Create train/val split (70:30):
       python create_ucla_split.py --video-dir data/multiview_action_videos --output-dir data/ucla_splits
    
    2. Process training videos (using config):
       python main_multipart_yolo_ucla.py --config config/ucla/yolo_pose.yaml --split train
    
    3. Process validation videos (using config):
       python main_multipart_yolo_ucla.py --config config/ucla/yolo_pose.yaml --split val
    
    Or use command line arguments directly (overrides config):
       python main_multipart_yolo_ucla.py --video-dir data/multiview_action_videos --output-dir data/ucla_yolo/train --split train --split-file data/ucla_splits/train_split.json
"""

from __future__ import print_function

import argparse
import os
import glob
import json
import re
import pickle
import yaml
from tqdm import tqdm
import numpy as np

from yolo_pose.yolo_pose_detector import YOLOPoseDetector


def extract_label_from_filename(filename: str) -> int:
    """
    Extract action label from UCLA filename format: a{action}_s{subject}_e{episode}_v{view}
    
    Args:
        filename: Video filename (e.g., 'a05_s04_e02_v03.mp4')
    
    Returns:
        Label (1-indexed action number)
    """
    match = re.match(r'a(\d+)_', filename)
    if match:
        action = int(match.group(1))
        return action
    return 0  # Default to 0 if pattern doesn't match


def process_videos(video_dir: str, output_dir: str, split: str, 
                  yolo_model_path: str = 'yolo11n-pose.pt',
                  yolo_conf_threshold: float = 0.25,
                  yolo_device: str = None,
                  yolo_tracking_strategy: str = 'largest_bbox',
                  overwrite: bool = False,
                  split_file: str = None):
    """
    Process all videos in a directory and save extracted keypoints.
    
    Args:
        video_dir: Root directory containing video files (or action class folders)
        output_dir: Directory to save extracted keypoints
        split: 'train' or 'val'
        yolo_model_path: Path to YOLO model weights
        yolo_conf_threshold: Confidence threshold for YOLO detection
        yolo_device: Device for YOLO inference ('cuda', 'cpu', or None for auto)
        yolo_tracking_strategy: Strategy for person selection
        overwrite: Whether to overwrite existing files
        split_file: Path to JSON file with split information (optional)
    """
    # Load split file if provided
    video_list = []
    if split_file and os.path.exists(split_file):
        print(f"Loading split information from {split_file}")
        with open(split_file, 'r') as f:
            split_data = json.load(f)
        
        # Extract video paths from split data
        for item in split_data:
            video_path = os.path.join(video_dir, item['video_path'])
            if os.path.exists(video_path):
                # Use full_file_name if available, otherwise fallback to file_name
                full_file_name = item.get('full_file_name', item['file_name'])
                video_list.append({
                    'path': video_path,
                    'name': item['file_name'],
                    'full_file_name': full_file_name,  # e.g., a06_v03_s07_e00
                    'label': item['label']
                })
            else:
                print(f"Warning: Video not found: {video_path}")
        
        print(f"Found {len(video_list)} videos from split file")
    else:
        # Fallback: find all video files in directory
        if not os.path.isdir(video_dir):
            raise ValueError(f"Video directory does not exist: {video_dir}")
        
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_dir, '**', ext), recursive=True))
            video_files.extend(glob.glob(os.path.join(video_dir, '**', ext.upper()), recursive=True))
        
        if not video_files:
            print(f"Warning: No video files found in {video_dir}")
            return
        
        video_files.sort()
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            # Try to extract label from path
            # Check if path contains action directory (e.g., .../a06/...)
            path_parts = video_path.split(os.sep)
            action_dir = None
            for part in path_parts:
                if part.startswith('a') and len(part) == 3 and part[1:].isdigit():
                    action_dir = part
                    break
            
            label = extract_label_from_filename(video_name)
            if action_dir:
                full_file_name = f"{action_dir}_{video_name}"
            else:
                full_file_name = video_name
            
            video_list.append({
                'path': video_path,
                'name': video_name,
                'full_file_name': full_file_name,  # e.g., a06_v03_s07_e00
                'label': label
            })
        
        print(f"Found {len(video_list)} video files in {video_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize YOLO detector
    print(f"Loading YOLO model from {yolo_model_path}...")
    detector = YOLOPoseDetector(
        model_path=yolo_model_path,
        conf_threshold=yolo_conf_threshold,
        device=yolo_device,
        tracking_strategy=yolo_tracking_strategy
    )
    print("YOLO model loaded successfully.")
    
    # Process each video
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    data_dict = []
    
    for video_item in tqdm(video_list, desc=f"Processing {split} videos"):
        video_path = video_item['path']
        video_name = video_item['name']
        full_file_name = video_item.get('full_file_name', video_name)  # Use full_file_name if available
        label = video_item['label']
        
        # Output file path using full_file_name (e.g., a06_v03_s07_e00.json)
        output_file = os.path.join(output_dir, f"{full_file_name}.json")
        
        # Skip if already exists and not overwriting
        if os.path.exists(output_file) and not overwrite:
            print(f"Skipping {video_name} (already exists)")
            skipped_count += 1
            # Still add to data_dict for consistency
            try:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                data_dict.append({
                    "file_name": video_name,
                    "full_file_name": full_file_name,
                    "length": len(data['skeletons']),
                    "label": label
                })
            except:
                pass
            continue
        
        # Process video
        try:
            keypoints_list, metadata = detector.detect_video(video_path)
            
            # Convert keypoints to format similar to UCLA JSON structure
            # UCLA format: skeletons is a list of frames, each frame is (20, 3) for 20 joints
            # YOLO format: list of (17, 3) arrays [x, y, confidence]
            # We'll save as (T, 17, 3) and pad to (T, 20, 3) if needed, or just use 17
            
            # Convert to numpy array: (T, 17, 3)
            skeletons_array = np.array(keypoints_list)  # (T, 17, 3)
            
            # Convert to list format for JSON serialization
            skeletons_list = skeletons_array.tolist()
            
            # Save as JSON (similar to UCLA format)
            output_data = {
                "skeletons": skeletons_list,  # List of frames, each frame is list of 17 joints [x, y, conf]
                "metadata": {
                    "num_frames": len(keypoints_list),
                    "num_joints": 17,
                    "video_path": video_path,
                    "label": label,
                    "fps": metadata.get('fps', 0),
                    "frame_count": metadata.get('frame_count', len(keypoints_list)),
                    "width": metadata.get('width', 0),
                    "height": metadata.get('height', 0)
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Add to data_dict
            data_dict.append({
                "file_name": video_name,
                "full_file_name": full_file_name,  # Include full_file_name in data_dict
                "length": len(keypoints_list),
                "label": label
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            error_count += 1
            continue
    
    # Save data_dict (list of all processed files with metadata)
    data_dict_file = os.path.join(output_dir, f"{split}_data_dict.json")
    with open(data_dict_file, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(video_list)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Data dict saved to: {data_dict_file}")


def get_parser():
    parser = argparse.ArgumentParser(
        description='Extract YOLO pose keypoints from UCLA video files')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (e.g., config/ucla/yolo_pose.yaml)')
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default=None,
        help='Directory containing video files (overrides config)')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save extracted keypoints (overrides config)')
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val'],
        required=True,
        help='Split name: train or val')
    
    parser.add_argument(
        '--yolo-model-path',
        type=str,
        default='yolo11n-pose.pt',
        help='Path to YOLO model weights (default: yolo11n-pose.pt)')
    
    parser.add_argument(
        '--yolo-conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold for YOLO detection (default: 0.25)')
    
    parser.add_argument(
        '--yolo-device',
        type=str,
        default=None,
        help='Device for YOLO inference: cuda, cpu, or None for auto (default: None)')
    
    parser.add_argument(
        '--yolo-tracking-strategy',
        type=str,
        default='largest_bbox',
        choices=['largest_bbox', 'highest_conf', 'track_id'],
        help='Strategy for person selection (default: largest_bbox)')
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files')
    
    parser.add_argument(
        '--split-file',
        type=str,
        default=None,
        help='Path to JSON file with split information (from create_ucla_split.py)')
    
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Load config file if provided
    config_data = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
    
    # Get preprocessing parameters from config or command line
    prep_config = config_data.get('preprocessing', {})
    
    # Determine video_dir, output_dir, split_file from config or args
    video_dir = args.video_dir or prep_config.get('video_dir')
    split_file_dir = prep_config.get('split_file_dir', 'data/ucla_splits')
    
    # Determine split file based on split argument
    if args.split == 'train':
        split_file = args.split_file or prep_config.get('train_split_file') or os.path.join(split_file_dir, 'train_split.json')
        output_dir = args.output_dir or os.path.join(prep_config.get('output_dir', 'data/ucla_yolo'), 'train')
    elif args.split == 'val':
        split_file = args.split_file or prep_config.get('val_split_file') or os.path.join(split_file_dir, 'val_split.json')
        output_dir = args.output_dir or os.path.join(prep_config.get('output_dir', 'data/ucla_yolo'), 'val')
    else:
        output_dir = args.output_dir or prep_config.get('output_dir', 'data/ucla_yolo')
        split_file = args.split_file
    
    # Get YOLO parameters from config or args
    yolo_model_path = args.yolo_model_path or prep_config.get('yolo_model_path', 'yolo11n-pose.pt')
    yolo_conf_threshold = args.yolo_conf_threshold if args.yolo_conf_threshold is not None else prep_config.get('yolo_conf_threshold', 0.25)
    yolo_device = args.yolo_device if args.yolo_device is not None else prep_config.get('yolo_device')
    yolo_tracking_strategy = args.yolo_tracking_strategy or prep_config.get('yolo_tracking_strategy', 'largest_bbox')
    # overwrite: use command line if provided, otherwise use config
    overwrite = prep_config.get('overwrite', False)
    if args.overwrite:  # If --overwrite flag is set, use it
        overwrite = True
    
    if not video_dir:
        raise ValueError("video_dir must be provided either via --video-dir or in config file")
    if not output_dir:
        raise ValueError("output_dir must be provided either via --output-dir or in config file")
    
    process_videos(
        video_dir=video_dir,
        output_dir=output_dir,
        split=args.split,
        yolo_model_path=yolo_model_path,
        yolo_conf_threshold=yolo_conf_threshold,
        yolo_device=yolo_device,
        yolo_tracking_strategy=yolo_tracking_strategy,
        overwrite=overwrite,
        split_file=split_file
    )
