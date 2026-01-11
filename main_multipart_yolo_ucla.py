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
import inspect
import os
import glob
import json
import re
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
import csv
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

# UCLA-specific text prompts
classes, num_text_aug, text_dict = text_prompt_openai_pasta_pool_4part_ucla()
text_list = text_prompt_openai_random_ucla()

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = torch.cuda.amp.GradScaler()

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


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
                  split_file: str = None,
                  data_json_path: str = None,
                  debug: bool = False):
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
        split_file: Path to JSON file with split information (optional, deprecated - use data_json_path)
        data_json_path: Path to data.json file (optional, if provided will update it after processing)
        debug: If True, process only 1/5 of videos for faster testing
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
    
    # Debug mode: reduce to 1/5 of videos
    if debug and len(video_list) > 5:
        original_count = len(video_list)
        # Take every 5th video (approximately 1/5)
        video_list = video_list[::5]
        print(f"DEBUG MODE: Reduced video list from {original_count} to {len(video_list)} videos (1/5)")
    
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
    
    # Update data.json if path provided
    if data_json_path:
        print(f"\nUpdating data.json: {data_json_path}")
        # Load existing data.json or create new
        if os.path.exists(data_json_path):
            with open(data_json_path, 'r') as f:
                data_json = json.load(f)
        else:
            data_json = {"train": [], "val": []}
        
        # Update the split section
        data_json[split] = []
        for item in data_dict:
            full_file_name = item.get('full_file_name', item['file_name'])
            json_path = os.path.join(output_dir, f"{full_file_name}.json")
            label = item['label']
            if isinstance(label, int) and label > 0:
                label = label - 1  # Convert to 0-indexed
            
            data_json[split].append({
                "json_path": json_path,
                "label": label,
                "full_file_name": full_file_name,
                "file_name": item['file_name']
            })
        
        # Save updated data.json
        os.makedirs(os.path.dirname(data_json_path), exist_ok=True)
        with open(data_json_path, 'w') as f:
            json.dump(data_json, f, indent=2)
        print(f"  Updated {split} section with {len(data_json[split])} entries")


def generate_data_json_from_splits(train_split_file: str, val_split_file: str, 
                                   output_dir: str, data_json_path: str):
    """
    Generate data.json file from existing split files.
    
    Args:
        train_split_file: Path to train_split.json
        val_split_file: Path to val_split.json
        output_dir: Base output directory for JSON files (e.g., data/yolo_ucla_json)
        data_json_path: Path where data.json should be saved
    """
    data_json = {"train": [], "val": []}
    
    # Process train split
    if os.path.exists(train_split_file):
        print(f"Loading train split from {train_split_file}")
        with open(train_split_file, 'r') as f:
            train_split = json.load(f)
        
        train_output_dir = os.path.join(output_dir, 'train')
        for item in train_split:
            full_file_name = item.get('full_file_name', item['file_name'])
            json_path = os.path.join(train_output_dir, f"{full_file_name}.json")
            # Convert label to 0-indexed if needed
            label = item['label']
            if isinstance(label, int) and label > 0:
                label = label - 1
            
            data_json["train"].append({
                "json_path": json_path,
                "label": label,
                "full_file_name": full_file_name,
                "file_name": item['file_name']
            })
        print(f"  Added {len(data_json['train'])} train entries")
    else:
        print(f"Warning: Train split file not found: {train_split_file}")
    
    # Process val split
    if os.path.exists(val_split_file):
        print(f"Loading val split from {val_split_file}")
        with open(val_split_file, 'r') as f:
            val_split = json.load(f)
        
        val_output_dir = os.path.join(output_dir, 'val')
        for item in val_split:
            full_file_name = item.get('full_file_name', item['file_name'])
            json_path = os.path.join(val_output_dir, f"{full_file_name}.json")
            # Convert label to 0-indexed if needed
            label = item['label']
            if isinstance(label, int) and label > 0:
                label = label - 1
            
            data_json["val"].append({
                "json_path": json_path,
                "label": label,
                "full_file_name": full_file_name,
                "file_name": item['file_name']
            })
        print(f"  Added {len(data_json['val'])} val entries")
    else:
        print(f"Warning: Val split file not found: {val_split_file}")
    
    # Save data.json
    os.makedirs(os.path.dirname(data_json_path), exist_ok=True)
    with open(data_json_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    
    print(f"\nGenerated data.json: {data_json_path}")
    print(f"  Train entries: {len(data_json['train'])}")
    print(f"  Val entries: {len(data_json['val'])}")
    
    return data_json


def generate_data_json_from_json_files(output_dir: str, data_json_path: str):
    """
    Generate data.json file by scanning existing JSON files in output directories.
    
    Args:
        output_dir: Base output directory containing train/ and val/ subdirectories
        data_json_path: Path where data.json should be saved
    """
    data_json = {"train": [], "val": []}
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # Process train directory
    if os.path.isdir(train_dir):
        train_json_files = glob.glob(os.path.join(train_dir, '*.json'))
        # Filter out data_dict files
        train_json_files = [f for f in train_json_files if not f.endswith('_data_dict.json')]
        
        for json_file in train_json_files:
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                full_file_name = os.path.splitext(os.path.basename(json_file))[0]
                # Extract label from metadata or filename
                label = json_data.get('metadata', {}).get('label', 0)
                if isinstance(label, int) and label > 0:
                    label = label - 1  # Convert to 0-indexed
                
                # Extract file_name from full_file_name (remove action prefix if present)
                file_name = full_file_name
                if '_' in full_file_name:
                    parts = full_file_name.split('_', 1)
                    if len(parts) > 1 and parts[0].startswith('a'):
                        file_name = parts[1]
                
                data_json["train"].append({
                    "json_path": json_file,
                    "label": label,
                    "full_file_name": full_file_name,
                    "file_name": file_name
                })
            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}")
        
        print(f"  Found {len(data_json['train'])} train JSON files")
    
    # Process val directory
    if os.path.isdir(val_dir):
        val_json_files = glob.glob(os.path.join(val_dir, '*.json'))
        # Filter out data_dict files
        val_json_files = [f for f in val_json_files if not f.endswith('_data_dict.json')]
        
        for json_file in val_json_files:
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                full_file_name = os.path.splitext(os.path.basename(json_file))[0]
                # Extract label from metadata or filename
                label = json_data.get('metadata', {}).get('label', 0)
                if isinstance(label, int) and label > 0:
                    label = label - 1  # Convert to 0-indexed
                
                # Extract file_name from full_file_name
                file_name = full_file_name
                if '_' in full_file_name:
                    parts = full_file_name.split('_', 1)
                    if len(parts) > 1 and parts[0].startswith('a'):
                        file_name = parts[1]
                
                data_json["val"].append({
                    "json_path": json_file,
                    "label": label,
                    "full_file_name": full_file_name,
                    "file_name": file_name
                })
            except Exception as e:
                print(f"Warning: Could not process {json_file}: {e}")
        
        print(f"  Found {len(data_json['val'])} val JSON files")
    
    # Save data.json
    os.makedirs(os.path.dirname(data_json_path), exist_ok=True)
    with open(data_json_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    
    print(f"\nGenerated data.json: {data_json_path}")
    print(f"  Train entries: {len(data_json['train'])}")
    print(f"  Val entries: {len(data_json['val'])}")
    
    return data_json


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True 

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_training_parser():
    """Get parser for training mode."""
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
        help='if ture, the classification score will be stored')
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


class Processor():
    """ 
    Processor for Skeleton-based Action Recognition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Ensure debug key exists in train_feeder_args
            if 'debug' not in arg.train_feeder_args:
                arg.train_feeder_args['debug'] = False
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
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
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        if torch.cuda.is_available():
            self.output_device = output_device
        else:
            self.output_device = None
            print("WARNING: CUDA not available, using CPU")

        # Check if model is set
        if self.arg.model is None:
            raise ValueError(f"Model is None. Please check your config file. Available attributes: {[k for k in dir(self.arg) if not k.startswith('_')]}")
        
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        
        # Filter out 'head' from model_args as it's used by Processor, not Model
        model_args_for_model = {k: v for k, v in self.arg.model_args.items() if k != 'head'}
        self.model = Model(**model_args_for_model)
        self.loss_ce = nn.CrossEntropyLoss().cuda(output_device)
        self.loss = KLLoss().cuda(output_device)

        self.model_text_dict = nn.ModuleDict()
        
        # Check if head is in model_args
        if 'head' not in self.arg.model_args:
            raise KeyError("'head' key not found in model_args. Please add 'head: [\"ViT-B/32\"]' to model_args in your config file.")
        
        for name in self.arg.model_args['head']:
            clip_device = torch.device(f'cuda:{self.output_device}' if torch.cuda.is_available() else 'cpu')
            model_, preprocess = clip.load(name, clip_device)
            del model_.visual
            model_text = TextCLIP(model_)
            model_text = model_text.cuda(self.output_device)
            self.model_text_dict[name] = model_text

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

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

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
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
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

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
                loss = loss_ce + self.arg.loss_alpha * sum(loss_te_list) / len(loss_te_list)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            if self.global_step % 100 == 0:
                torch.cuda.empty_cache()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
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
            process = tqdm(self.data_loader[ln], ncols=40)
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
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            start_epoch = 0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])

            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
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


def run_training(config_path: str, debug: bool = False):
    """
    Run training mode using Processor.
    
    Args:
        config_path: Path to config file with training configuration
        debug: If True, enable debug mode (reduces dataset size)
    """
    # Create parser
    parser = get_training_parser()
    
    # Load config file
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get training config
    training_config = config_data.get('training', {})
    
    if not training_config:
        raise ValueError(f"No 'training' section found in config file: {config_path}")
    
    # Create a temporary config file with training section flattened
    import tempfile
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    
    # Flatten training config for parser (remove 'training' nesting)
    flattened_config = {}
    for key, value in training_config.items():
        flattened_config[key] = value
    
    # Enable debug mode if requested - ensure debug is set in feeder args
    if debug:
        if 'train_feeder_args' not in flattened_config:
            flattened_config['train_feeder_args'] = {}
        if 'test_feeder_args' not in flattened_config:
            flattened_config['test_feeder_args'] = {}
        flattened_config['train_feeder_args']['debug'] = True
        flattened_config['test_feeder_args']['debug'] = True
        print("DEBUG MODE: Enabled debug mode in feeder (will use 1/5 of data)")
    
    # Verify required fields are present
    if 'model' not in flattened_config:
        raise ValueError("'model' field not found in training config. Please check your config file.")
    if 'feeder' not in flattened_config:
        raise ValueError("'feeder' field not found in training config. Please check your config file.")
    
    yaml.dump(flattened_config, temp_config, default_flow_style=False)
    temp_config.close()
    
    try:
        # Load config file first
        with open(temp_config.name, 'r') as f:
            default_arg = yaml.safe_load(f)
        
        # Verify required fields exist in config
        if 'model' not in default_arg:
            raise ValueError(f"'model' field not found in training config. Available keys: {list(default_arg.keys())}")
        if 'feeder' not in default_arg:
            raise ValueError(f"'feeder' field not found in training config. Available keys: {list(default_arg.keys())}")
        
        # Parse once to get parser structure
        p = parser.parse_args(['--config', temp_config.name])
        
        # Check that all keys in config are valid parser arguments
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print(f'WARNING: Config key "{k}" not found in parser arguments. This might be handled by DictAction.')
        
        # Set defaults from config (same approach as main_multipart_ucla.py)
        parser.set_defaults(**default_arg)
        
        # Parse again to get final args with defaults applied
        arg = parser.parse_args(['--config', temp_config.name])
        
        # Manually ensure critical fields are set (in case parser didn't apply them)
        if not arg.model and 'model' in default_arg:
            arg.model = default_arg['model']
        if not arg.feeder and 'feeder' in default_arg:
            arg.feeder = default_arg['feeder']
        
        # Also ensure nested dicts are properly set
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
        
        # Verify critical fields are set
        if not arg.model:
            raise ValueError(f"Model not set after parsing. Config had: {default_arg.get('model', 'NOT FOUND')}, arg.model: {arg.model}")
        if not arg.feeder:
            raise ValueError(f"Feeder not set after parsing. Config had: {default_arg.get('feeder', 'NOT FOUND')}, arg.feeder: {arg.feeder}")
        
        print(f"Loaded config - Model: {arg.model}, Feeder: {arg.feeder}, Work dir: {arg.work_dir}")
        
        # Initialize and run
        init_seed(getattr(arg, 'seed', 0))
        processor = Processor(arg)
        processor.start()
    finally:
        # Clean up temp file
        if os.path.exists(temp_config.name):
            os.unlink(temp_config.name)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Extract YOLO pose keypoints from UCLA video files or run training')
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (e.g., config/yolo_ucla/data_config.yaml)')
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['process', 'train'],
        default=None,
        help='Mode: process (extract keypoints) or train (run training). Auto-detected if not specified.')
    
    parser.add_argument(
        '--video-dir',
        type=str,
        default=None,
        help='Directory containing video files (overrides config, for process mode)')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save extracted keypoints (overrides config, for process mode)')
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val'],
        default=None,
        help='Split name: train or val (for process mode, required if not using data.json)')
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug mode: reduce dataset to 1/5 size for faster testing')
    
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
    
    # Determine mode
    mode = args.mode
    if not mode:
        # Auto-detect mode: if training section exists, use train mode, otherwise process
        if config_data.get('training'):
            mode = 'train'
        else:
            mode = 'process'
    
    if mode == 'train':
        # Training mode
        if not args.config:
            raise ValueError("--config is required for training mode")
        debug_mode = getattr(args, 'debug', False)
        if debug_mode:
            print("Running in TRAINING mode (DEBUG: dataset reduced to 1/5)...")
        else:
            print("Running in TRAINING mode...")
        run_training(args.config, debug=debug_mode)
    
    elif mode == 'process':
        # Processing mode
        data_config = config_data.get('data_config', {})
        prep_config = config_data.get('processing', {})
        data_json_path = data_config.get('data_json')
        
        # Determine video_dir, output_dir from config or args
        video_dir = args.video_dir or prep_config.get('video_dir')
        output_dir_base = args.output_dir or prep_config.get('output_dir', 'data/yolo_ucla_json')
        
        # Get split - either from args or process both train and val
        if args.split:
            splits_to_process = [args.split]
        elif data_json_path and os.path.exists(data_json_path):
            # If data.json exists, we can determine splits from it
            with open(data_json_path, 'r') as f:
                data_json = json.load(f)
            splits_to_process = []
            if data_json.get('train'):
                splits_to_process.append('train')
            if data_json.get('val'):
                splits_to_process.append('val')
            if not splits_to_process:
                splits_to_process = ['train', 'val']  # Default to both
        else:
            # Default: process both splits
            splits_to_process = ['train', 'val']
        
        # Get YOLO parameters from config or args
        yolo_model_path = getattr(args, 'yolo_model_path', None) or prep_config.get('yolo_model_path', 'yolo11n-pose.pt')
        yolo_conf_threshold = getattr(args, 'yolo_conf_threshold', None)
        if yolo_conf_threshold is None:
            yolo_conf_threshold = prep_config.get('yolo_conf_threshold', 0.25)
        yolo_device = getattr(args, 'yolo_device', None)
        if yolo_device is None:
            yolo_device = prep_config.get('yolo_device')
        yolo_tracking_strategy = getattr(args, 'yolo_tracking_strategy', None) or prep_config.get('yolo_tracking_strategy', 'largest_bbox')
        overwrite = prep_config.get('overwrite', False)
        if getattr(args, 'overwrite', False):
            overwrite = True
        
        if not video_dir:
            raise ValueError("video_dir must be provided either via --video-dir or in config file")
        
        # Get debug mode
        debug_mode = getattr(args, 'debug', False)
        if debug_mode:
            print("DEBUG MODE: Will process only 1/5 of videos for faster testing")
        
        # Try to load data.json to get split information
        split_file_dir = 'data/ucla_splits'  # Default
        if data_json_path and os.path.exists(data_json_path):
            # Use data.json to determine which videos to process
            with open(data_json_path, 'r') as f:
                data_json = json.load(f)
            
            # Process each split
            for split in splits_to_process:
                print(f"\n{'='*60}")
                print(f"Processing {split} split...")
                print(f"{'='*60}")
                
                output_dir = os.path.join(output_dir_base, split)
                split_data = data_json.get(split, [])
                
                # Debug mode: reduce to 1/5
                if debug_mode and len(split_data) > 5:
                    original_count = len(split_data)
                    split_data = split_data[::5]  # Take every 5th item
                    print(f"DEBUG MODE: Reduced {split} split from {original_count} to {len(split_data)} items (1/5)")
                
                if split_data:
                    # Extract video paths from data.json
                    video_list = []
                    for item in split_data:
                        json_path = item.get('json_path', '')
                        # If JSON file already exists, skip (unless overwrite)
                        if os.path.exists(json_path) and not overwrite:
                            continue
                        
                        # Try to find video file from json_path
                        # json_path format: data/yolo_ucla_json/train/a01_v01_s01_e00.json
                        # Need to find original video
                        full_file_name = item.get('full_file_name', '')
                        # Try to construct video path
                        # Assume videos are in video_dir with action folders
                        if full_file_name.startswith('a'):
                            action_dir = full_file_name.split('_')[0]
                            video_name = '_'.join(full_file_name.split('_')[1:])
                            video_path = os.path.join(video_dir, action_dir, f"{video_name}.avi")
                            if not os.path.exists(video_path):
                                # Try other extensions
                                for ext in ['.mp4', '.mov', '.mkv']:
                                    alt_path = os.path.join(video_dir, action_dir, f"{video_name}{ext}")
                                    if os.path.exists(alt_path):
                                        video_path = alt_path
                                        break
                            
                            if os.path.exists(video_path):
                                video_list.append({
                                    'path': video_path,
                                    'name': video_name,
                                    'full_file_name': full_file_name,
                                    'label': item.get('label', 0)
                                })
                    
                    if video_list:
                        # Process videos
                        process_videos(
                            video_dir=video_dir,
                            output_dir=output_dir,
                            split=split,
                            yolo_model_path=yolo_model_path,
                            yolo_conf_threshold=yolo_conf_threshold,
                            yolo_device=yolo_device,
                            yolo_tracking_strategy=yolo_tracking_strategy,
                            overwrite=overwrite,
                            split_file=None,  # Not using split_file when using data.json
                            data_json_path=data_json_path,
                            debug=debug_mode
                        )
                    else:
                        print(f"No videos to process for {split} split (all JSON files exist)")
                else:
                    print(f"No data found in data.json for {split} split")
        else:
            # Fallback to old method using split files
            print("Warning: data.json not found, using split files (legacy mode)")
            split_file_dir = 'data/ucla_splits'
            
            for split in splits_to_process:
                print(f"\n{'='*60}")
                print(f"Processing {split} split...")
                print(f"{'='*60}")
                
                output_dir = os.path.join(output_dir_base, split)
                split_file = os.path.join(split_file_dir, f'{split}_split.json')
                
                process_videos(
                    video_dir=video_dir,
                    output_dir=output_dir,
                    split=split,
                    yolo_model_path=yolo_model_path,
                    yolo_conf_threshold=yolo_conf_threshold,
                    yolo_device=yolo_device,
                    yolo_tracking_strategy=yolo_tracking_strategy,
                    overwrite=overwrite,
                    split_file=split_file if os.path.exists(split_file) else None,
                    data_json_path=data_json_path,
                    debug=debug_mode
                )
        
        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
