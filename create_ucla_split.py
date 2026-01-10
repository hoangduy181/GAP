#!/usr/bin/env python
"""
Create train/val split for UCLA multiview action videos.

Scans the multiview_action_videos directory structure and creates a 70:30 train/val split.
Saves split information to JSON files that can be used by the preprocessing script.

Usage:
    python create_ucla_split.py --video-dir data/multiview_action_videos --output-dir data/ucla_splits
"""

from __future__ import print_function

import argparse
import os
import glob
import json
import re
import random
from collections import defaultdict


def extract_action_label(directory_name: str) -> int:
    """
    Extract action label from directory name (e.g., 'a01' -> 0, 'a02' -> 1).
    
    Args:
        directory_name: Directory name like 'a01', 'a02', etc.
    
    Returns:
        Label (0-indexed)
    """
    match = re.match(r'a(\d+)', directory_name)
    if match:
        action = int(match.group(1))
        return action - 1  # Convert to 0-indexed (a01 -> 0, a02 -> 1, etc.)
    return -1


def create_split(video_dir: str, output_dir: str, train_ratio: float = 0.7, seed: int = 42):
    """
    Create train/val split from video directory structure.
    
    Args:
        video_dir: Root directory containing action class folders (a01, a02, etc.)
        output_dir: Directory to save split files
        train_ratio: Ratio of training samples (default: 0.7 for 70:30 split)
        seed: Random seed for reproducibility
    """
    if not os.path.isdir(video_dir):
        raise ValueError(f"Video directory does not exist: {video_dir}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Find all action class directories
    action_dirs = sorted([d for d in os.listdir(video_dir) 
                         if os.path.isdir(os.path.join(video_dir, d)) and d.startswith('a')])
    
    print(f"Found {len(action_dirs)} action class directories")
    
    # Collect all videos grouped by action class
    all_videos = defaultdict(list)
    
    for action_dir in action_dirs:
        action_path = os.path.join(video_dir, action_dir)
        label = extract_action_label(action_dir)
        
        if label < 0:
            print(f"Warning: Could not extract label from {action_dir}, skipping")
            continue
        
        # Find all video files in this action directory
        video_extensions = ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.flv', '*.wmv']
        videos = []
        for ext in video_extensions:
            videos.extend(glob.glob(os.path.join(action_path, ext)))
            videos.extend(glob.glob(os.path.join(action_path, ext.upper())))
        
        videos.sort()
        
        for video_path in videos:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            # Create full file name with action prefix (e.g., a06_v03_s07_e00)
            full_file_name = f"{action_dir}_{video_name}"
            # Create relative path from video_dir
            rel_path = os.path.relpath(video_path, video_dir)
            
            all_videos[label].append({
                'file_name': video_name,
                'full_file_name': full_file_name,  # e.g., a06_v03_s07_e00
                'video_path': rel_path,
                'full_path': video_path,
                'label': label,
                'action_dir': action_dir
            })
        
        print(f"  {action_dir} (label {label}): {len(videos)} videos")
    
    # Create train/val split (70:30)
    train_data = []
    val_data = []
    
    for label, videos in all_videos.items():
        # Shuffle videos for this class
        shuffled = videos.copy()
        random.shuffle(shuffled)
        
        # Split based on train_ratio
        n_train = int(len(shuffled) * train_ratio)
        train_videos = shuffled[:n_train]
        val_videos = shuffled[n_train:]
        
        train_data.extend(train_videos)
        val_data.extend(val_videos)
        
        print(f"  Class {label}: {len(train_videos)} train, {len(val_videos)} val")
    
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
        'num_classes': len(all_videos),
        'classes': sorted(all_videos.keys()),
        'videos_per_class': {label: len(videos) for label, videos in all_videos.items()},
        'train_per_class': {},
        'val_per_class': {}
    }
    
    # Count videos per class in train/val
    for label in all_videos.keys():
        summary['train_per_class'][label] = sum(1 for v in train_data if v['label'] == label)
        summary['val_per_class'][label] = sum(1 for v in val_data if v['label'] == label)
    
    summary_file = os.path.join(output_dir, 'split_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Split Summary:")
    print(f"  Total videos: {summary['total_videos']}")
    print(f"  Train videos: {summary['train_videos']} ({100*summary['train_videos']/summary['total_videos']:.1f}%)")
    print(f"  Val videos: {summary['val_videos']} ({100*summary['val_videos']/summary['total_videos']:.1f}%)")
    print(f"  Number of classes: {summary['num_classes']}")
    print(f"{'='*60}")
    
    return train_data, val_data, summary


def get_parser():
    parser = argparse.ArgumentParser(
        description='Create train/val split for UCLA multiview action videos')
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/multiview_action_videos',
        help='Root directory containing action class folders (default: data/multiview_action_videos)')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/ucla_splits',
        help='Directory to save split files (default: data/ucla_splits)')
    
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
