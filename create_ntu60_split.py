#!/usr/bin/env python
"""
Create train/val split for NTU RGB+D 60 video files.

Scans the video directory and creates a 70:30 train/val split.
Saves split information to JSON files that can be used by the preprocessing script.

NTU60 video naming convention: S{setup}C{camera}P{performer}R{repetition}A{action}_rgb.avi
Example: S001C001P001R001A001_rgb.avi

Usage:
    python create_ntu60_split.py --video-dir data/nturgb+d_rgb --output-dir data/ntu60_splits
"""

from __future__ import print_function

import argparse
import os
import glob
import json
import re
import random
from collections import defaultdict


def parse_ntu_filename(filename: str) -> dict:
    """
    Parse NTU60 video filename to extract components.
    
    Format: S{setup}C{camera}P{performer}R{repetition}A{action}_rgb.avi
    Example: S001C001P001R001A001_rgb.avi
    
    Args:
        filename: Video filename (with or without extension)
    
    Returns:
        Dictionary with setup, camera, performer, repetition, action, and label_action, label_subject
        Returns None if parsing fails
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    # Remove _rgb suffix if present
    name_without_ext = name_without_ext.replace('_rgb', '')
    
    # Match pattern: S001C001P001R001A001
    match = re.match(r'S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)', name_without_ext)
    if match:
        setup = int(match.group(1))
        camera = int(match.group(2))
        performer = int(match.group(3))  # Subject/Performer
        repetition = int(match.group(4))
        action = int(match.group(5))
        
        return {
            'setup': setup,
            'camera': camera,
            'performer': performer,
            'repetition': repetition,
            'action': action,
            'label_action': action - 1,  # 0-indexed (A001 -> 0, A002 -> 1, etc.)
            'label_subject': performer - 1  # 0-indexed (P001 -> 0, P002 -> 1, etc.)
        }
    
    return None


def create_split(video_dir: str, output_dir: str, train_ratio: float = 0.7, seed: int = 42):
    """
    Create train/val split from NTU60 video directory.
    
    Args:
        video_dir: Root directory containing video files
        output_dir: Directory to save split files
        train_ratio: Ratio of training samples (default: 0.7 for 70:30 split)
        seed: Random seed for reproducibility
    """
    if not os.path.isdir(video_dir):
        raise ValueError(f"Video directory does not exist: {video_dir}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all video files
    video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        video_files.extend(glob.glob(os.path.join(video_dir, ext.upper())))
    
    if not video_files:
        print(f"Warning: No video files found in {video_dir}")
        return
    
    video_files.sort()
    print(f"Found {len(video_files)} video files")
    
    # Parse all videos and group by action
    all_videos = defaultdict(list)
    skipped_count = 0
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Parse filename
        parsed = parse_ntu_filename(video_name)
        if parsed is None:
            print(f"Warning: Could not parse filename {video_name}, skipping")
            skipped_count += 1
            continue
        
        # Create relative path from video_dir
        rel_path = os.path.relpath(video_path, video_dir)
        
        video_data = {
            'video_name': video_name,
            'video_path': rel_path,
            'full_path': video_path,
            'label_action': parsed['label_action'],
            'label_subject': parsed['label_subject'],
            'setup': parsed['setup'],
            'camera': parsed['camera'],
            'performer': parsed['performer'],
            'repetition': parsed['repetition'],
            'action': parsed['action']
        }
        
        # Group by action label
        all_videos[parsed['label_action']].append(video_data)
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files that couldn't be parsed")
    
    print(f"Successfully parsed {len(video_files) - skipped_count} videos")
    print(f"Found {len(all_videos)} action classes")
    
    # Print statistics per action class
    for label_action in sorted(all_videos.keys()):
        videos = all_videos[label_action]
        print(f"  Action {label_action} (A{label_action+1:03d}): {len(videos)} videos")
    
    # Create train/val split (70:30)
    train_data = []
    val_data = []
    
    for label_action, videos in all_videos.items():
        # Shuffle videos for this action class
        shuffled = videos.copy()
        random.shuffle(shuffled)
        
        # Split based on train_ratio
        n_train = int(len(shuffled) * train_ratio)
        train_videos = shuffled[:n_train]
        val_videos = shuffled[n_train:]
        
        train_data.extend(train_videos)
        val_data.extend(val_videos)
        
        print(f"  Action {label_action}: {len(train_videos)} train, {len(val_videos)} val")
    
    # Shuffle final lists
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train split
    train_file = os.path.join(output_dir, 'train_split.json')
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"\nSaved train split ({len(train_data)} videos) to {train_file}")
    
    # Save val split
    val_file = os.path.join(output_dir, 'val_split.json')
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"Saved val split ({len(val_data)} videos) to {val_file}")
    
    # Save summary statistics
    summary = {
        'total_videos': len(train_data) + len(val_data),
        'train_videos': len(train_data),
        'val_videos': len(val_data),
        'train_ratio': train_ratio,
        'num_action_classes': len(all_videos),
        'action_classes': sorted(all_videos.keys()),
        'videos_per_action': {label: len(videos) for label, videos in all_videos.items()},
        'train_per_action': {},
        'val_per_action': {},
        'num_subjects': len(set(v['label_subject'] for v in train_data + val_data)),
        'subjects': sorted(set(v['label_subject'] for v in train_data + val_data))
    }
    
    # Count videos per action in train/val
    for label_action in all_videos.keys():
        summary['train_per_action'][label_action] = sum(1 for v in train_data if v['label_action'] == label_action)
        summary['val_per_action'][label_action] = sum(1 for v in val_data if v['label_action'] == label_action)
    
    summary_file = os.path.join(output_dir, 'split_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    # Create data.json with format: { "train": { "split": "train", "items": [...] }, "val": { "split": "val", "items": [...] } }
    data_json = {
        "splits": [
            {
                "split": "train",
                "items": train_data
            },
            {
                "split": "val",
                "items": val_data
            }
        ],
        "metadata": {
            "total_videos": summary['total_videos'],
            "train_videos": len(train_data),
            "val_videos": len(val_data),
            "train_ratio": train_ratio,
            "num_action_classes": summary['num_action_classes'],
            "num_subjects": summary['num_subjects'],
            "action_classes": summary['action_classes'],
            "subjects": summary['subjects']
        }
    }
    
    data_json_file = os.path.join(output_dir, 'data.json')
    with open(data_json_file, 'w') as f:
        json.dump(data_json, f, indent=2)
    print(f"Saved data.json ({len(train_data)} train, {len(val_data)} val) to {data_json_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Split Summary:")
    print(f"  Total videos: {summary['total_videos']}")
    print(f"  Train videos: {summary['train_videos']} ({100*summary['train_videos']/summary['total_videos']:.1f}%)")
    print(f"  Val videos: {summary['val_videos']} ({100*summary['val_videos']/summary['total_videos']:.1f}%)")
    print(f"  Number of action classes: {summary['num_action_classes']}")
    print(f"  Number of subjects: {summary['num_subjects']}")
    print(f"{'='*60}")
    
    return train_data, val_data, summary


def get_parser():
    parser = argparse.ArgumentParser(
        description='Create train/val split for NTU RGB+D 60 video files')
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/nturgb+d_rgb',
        help='Root directory containing video files (default: data/nturgb+d_rgb)')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ntu60_splits',
        help='Directory to save split files (default: data/ntu60_splits)')
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio of training samples (default: 0.7 for 70:30 split)')
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)')
    
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    create_split(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
