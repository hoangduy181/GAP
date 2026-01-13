"""
Feeder for NTU60 YOLO pose detection from video files or preprocessed JSON files.

Supports loading from:
1. data.json format (with splits structure)
2. Direct JSON files in directory
3. Video files (with on-the-fly YOLO detection)
"""

import numpy as np
import os
import glob
import json
import re
from torch.utils.data import Dataset
from typing import Optional, List, Dict

from feeders import tools
from yolo_pose.yolo_pose_detector import YOLOPoseDetector


class Feeder(Dataset):
    """
    Feeder for NTU60 YOLO pose detection from video files or preprocessed JSON files.
    
    Processes video files using YOLO pose detection or loads preprocessed skeleton JSON files,
    and converts to CTR-GCN format for action recognition.
    """
    
    def __init__(self, 
                 data_path: str,
                 label_path: Optional[str] = None,
                 p_interval: float = 1.0,
                 split: str = 'train',
                 random_choose: bool = False,
                 random_shift: bool = False,
                 random_move: bool = False,
                 random_rot: bool = False,
                 window_size: int = -1,
                 normalization: bool = False,
                 debug: bool = False,
                 use_mmap: bool = False,
                 bone: bool = False,
                 vel: bool = False,
                 repeat: int = 1,
                 # YOLO-specific parameters
                 yolo_model_path: str = 'yolo11n-pose.pt',
                 yolo_conf_threshold: float = 0.25,
                 yolo_device: Optional[str] = None,
                 yolo_tracking_strategy: str = 'largest_bbox',
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True,
                 # Data.json support
                 data_json_path: Optional[str] = None):
        """
        Initialize NTU60 YOLO Pose Feeder.
        
        Args:
            data_path: Path to directory containing JSON files, or video directory, or data.json file
            label_path: Path to label file (optional, can be inferred from data.json)
            split: 'train' or 'test'/'val'
            p_interval: Proportion of valid frames to use
            window_size: Target number of frames per sample (-1 for original length)
            normalization: Normalize coordinates
            debug: Use only first 100 samples
            bone: Convert to bone representation
            vel: Convert to velocity/motion representation
            repeat: Number of times to repeat each sample (for data augmentation)
            yolo_model_path: Path to YOLO model weights
            yolo_conf_threshold: Confidence threshold for YOLO detection
            yolo_device: Device for YOLO inference ('cuda', 'cpu', or None for auto)
            yolo_tracking_strategy: Strategy for person selection
            cache_dir: Directory to cache YOLO detections
            use_cache: Whether to use cached detections if available
            data_json_path: Path to data.json file (optional, will be auto-detected if not provided)
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.repeat = repeat
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.data_json_path = data_json_path
        
        # Ensure p_interval is a list
        if isinstance(p_interval, (int, float)):
            self.p_interval = [p_interval]
        else:
            self.p_interval = p_interval
        
        # Initialize YOLO detector (only if needed for video processing)
        self.yolo_detector = None
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_device = yolo_device
        self.yolo_tracking_strategy = yolo_tracking_strategy
        
        # Load data
        self.load_data()
        
        if normalization:
            self.get_mean_map()
    
    def load_data(self):
        """
        Load data from data.json, JSON files, or video files.
        
        Priority:
        1. Explicit data_json_path
        2. data_path as data.json file
        3. data_path directory with JSON files
        4. data_path directory with video files (requires YOLO)
        """
        # Priority 1: Use explicitly provided data_json_path
        if self.data_json_path and os.path.exists(self.data_json_path):
            self._load_from_data_json(self.data_json_path)
            return
        
        # Priority 2: Check if data_path is a data.json file
        if os.path.isfile(self.data_path) and self.data_path.endswith('data.json'):
            self._load_from_data_json(self.data_path)
            return
        
        # Priority 3: Check if data_path is a directory with JSON files
        if os.path.isdir(self.data_path):
            # Check for data.json in directory or parent
            potential_data_json = os.path.join(self.data_path, 'data.json')
            if os.path.exists(potential_data_json):
                self._load_from_data_json(potential_data_json)
                return
            
            # Check parent directory
            parent_data_json = os.path.join(os.path.dirname(self.data_path), 'data.json')
            if os.path.exists(parent_data_json):
                self._load_from_data_json(parent_data_json)
                return
            
            # Check sibling directories
            parent_dir = os.path.dirname(self.data_path)
            for sibling_dir in ['ntu60_splits', 'splits']:
                sibling_data_json = os.path.join(parent_dir, sibling_dir, 'data.json')
                if os.path.exists(sibling_data_json):
                    self._load_from_data_json(sibling_data_json)
                    return
            
            # Try loading JSON files directly from directory
            json_files = glob.glob(os.path.join(self.data_path, '*.json'))
            json_files = [f for f in json_files if not f.endswith('_data_dict.json')]
            if json_files:
                self._load_from_json_files(json_files)
                return
            
            # Fallback: try to load from video files (requires YOLO)
            video_files = []
            for ext in ['*.avi', '*.mp4', '*.mov', '*.mkv']:
                video_files.extend(glob.glob(os.path.join(self.data_path, ext)))
            if video_files:
                print("Warning: No JSON files found, will process videos on-the-fly (slow)")
                self._load_from_video_files(video_files)
                return
        
        raise ValueError(f"No valid data source found in {self.data_path}")
    
    def _load_from_data_json(self, data_json_path: str):
        """
        Load data from data.json file with splits structure.
        
        Expected format:
        {
          "splits": [
            {
              "split": "train",
              "items": [
                {
                  "video_name": "S001C001P001R001A001_rgb",
                  "video_path": "S001C001P001R001A001_rgb.avi",
                  "json_path": "data/ntu60_json/train/S001C001P001R001A001_rgb.json",
                  "label_action": 0,
                  "label_subject": 0,
                  ...
                },
                ...
              ]
            },
            ...
          ]
        }
        """
        print(f"Loading from data.json: {data_json_path}")
        
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
        
        # Find the appropriate split
        split_data = None
        for split_info in data_json.get('splits', []):
            if split_info.get('split') == self.split or \
               (self.split == 'test' and split_info.get('split') == 'val'):
                split_data = split_info.get('items', [])
                break
        
        if not split_data:
            raise ValueError(f"No {self.split} split found in data.json")
        
        # Debug mode: limit to first 100 samples
        if self.debug:
            original_count = len(split_data)
            split_data = split_data[:100]
            print(f"DEBUG MODE: Reduced dataset from {original_count} to {len(split_data)} samples (first 100)")
        
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, item in enumerate(split_data):
            json_path = item.get('json_path')
            video_name = item.get('video_name', '')
            label_action = item.get('label_action', 0)
            
            # If json_path is provided and exists, load from JSON
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        skeleton_data = json.load(f)
                    
                    skeletons = skeleton_data.get('skeletons', [])
                    if not skeletons:
                        print(f"Warning: No skeletons in {json_path}, skipping")
                        continue
                    
                    # Convert to numpy array: (T, 17, 3)
                    keypoints_list = [np.array(frame) for frame in skeletons]
                    keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
                    
                    self.data_list.append(keypoints_array)
                    self.label_list.append(label_action)
                    self.sample_name.append(video_name)
                    self.indices.append(idx)
                    
                except Exception as e:
                    print(f"Warning: Failed to load {json_path}: {e}, skipping")
                    continue
            else:
                # Fallback: try to load from video file if available
                video_path = item.get('full_path') or item.get('video_path')
                if video_path and os.path.exists(video_path):
                    print(f"Warning: JSON not found for {video_name}, processing video on-the-fly")
                    try:
                        if self.yolo_detector is None:
                            self.yolo_detector = YOLOPoseDetector(
                                model_path=self.yolo_model_path,
                                conf_threshold=self.yolo_conf_threshold,
                                device=self.yolo_device,
                                tracking_strategy=self.yolo_tracking_strategy
                            )
                        
                        keypoints_list, metadata = self.yolo_detector.detect_video(video_path, return_frames=False)
                        keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
                        
                        self.data_list.append(keypoints_array)
                        self.label_list.append(label_action)
                        self.sample_name.append(video_name)
                        self.indices.append(idx)
                    except Exception as e:
                        print(f"Warning: Failed to process video {video_path}: {e}, skipping")
                        continue
        
        print(f"Loaded {len(self.data_list)} samples from data.json for {self.split} split")
    
    def _load_from_json_files(self, json_files: List[str]):
        """Load data from JSON files directly (legacy support)."""
        print(f"Loading from {len(json_files)} JSON files in directory")
        
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, json_file in enumerate(json_files):
            try:
                with open(json_file, 'r') as f:
                    skeleton_data = json.load(f)
                
                skeletons = skeleton_data.get('skeletons', [])
                metadata = skeleton_data.get('metadata', {})
                label_action = metadata.get('label_action', 0)
                video_name = metadata.get('video_name', os.path.splitext(os.path.basename(json_file))[0])
                
                if not skeletons:
                    continue
                
                keypoints_list = [np.array(frame) for frame in skeletons]
                keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
                
                self.data_list.append(keypoints_array)
                self.label_list.append(label_action)
                self.sample_name.append(video_name)
                self.indices.append(idx)
                
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        if self.debug:
            original_count = len(self.data_list)
            self.data_list = self.data_list[:100]
            self.label_list = self.label_list[:100]
            self.sample_name = self.sample_name[:100]
            self.indices = self.indices[:100]
            print(f"DEBUG MODE: Reduced dataset from {original_count} to {len(self.data_list)} samples")
        
        print(f"Loaded {len(self.data_list)} samples from JSON files")
    
    def _load_from_video_files(self, video_files: List[str]):
        """Load data from video files using YOLO (on-the-fly processing)."""
        print(f"Loading from {len(video_files)} video files (on-the-fly YOLO processing)")
        
        if self.yolo_detector is None:
            self.yolo_detector = YOLOPoseDetector(
                model_path=self.yolo_model_path,
                conf_threshold=self.yolo_conf_threshold,
                device=self.yolo_device,
                tracking_strategy=self.yolo_tracking_strategy
            )
        
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, video_path in enumerate(video_files):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Parse label from filename
            parsed = self._parse_ntu_filename(video_name)
            label_action = parsed['label_action'] if parsed else 0
            
            try:
                keypoints_list, metadata = self.yolo_detector.detect_video(video_path, return_frames=False)
                keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
                
                self.data_list.append(keypoints_array)
                self.label_list.append(label_action)
                self.sample_name.append(video_name)
                self.indices.append(idx)
            except Exception as e:
                print(f"Warning: Failed to process {video_path}: {e}")
                continue
        
        if self.debug:
            original_count = len(self.data_list)
            self.data_list = self.data_list[:100]
            self.label_list = self.label_list[:100]
            self.sample_name = self.sample_name[:100]
            self.indices = self.indices[:100]
            print(f"DEBUG MODE: Reduced dataset from {original_count} to {len(self.data_list)} samples")
        
        print(f"Loaded {len(self.data_list)} samples from video files")
    
    def _parse_ntu_filename(self, filename: str) -> Optional[dict]:
        """Parse NTU60 filename to extract action and subject labels."""
        name_without_ext = os.path.splitext(filename)[0]
        name_without_ext = name_without_ext.replace('_rgb', '')
        
        match = re.match(r'S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)', name_without_ext)
        if match:
            return {
                'label_action': int(match.group(5)) - 1,  # 0-indexed
                'label_subject': int(match.group(3)) - 1   # 0-indexed
            }
        return None
    
    def _yolo_to_ctrgcn_format(self, keypoints_list: List[np.ndarray]) -> np.ndarray:
        """
        Convert YOLO keypoints list to CTR-GCN format: (C=2, T, V=17, M=1)
        
        Args:
            keypoints_list: List of keypoint arrays, each (17, 3) [x, y, confidence]
        
        Returns:
            Array in CTR-GCN format: (2, T, 17, 1)
        """
        if not keypoints_list:
            return np.zeros((2, 1, 17, 1), dtype=np.float32)
        
        # Stack keypoints: (T, 17, 3)
        keypoints_stack = np.stack(keypoints_list, axis=0)
        
        # Extract x, y coordinates (ignore confidence): (T, 17, 2)
        keypoints_xy = keypoints_stack[:, :, :2]
        
        # Transpose to (2, T, 17)
        keypoints_ctrgcn = keypoints_xy.transpose(2, 0, 1)
        
        # Add M dimension: (2, T, 17, 1)
        keypoints_final = keypoints_ctrgcn[:, :, :, np.newaxis]
        
        return keypoints_final.astype(np.float32)
    
    def get_mean_map(self):
        """Calculate mean and std maps for normalization."""
        data = np.array(self.data_list)
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    
    def __len__(self):
        """Return dataset length (with repeat augmentation)."""
        return len(self.data_list) * self.repeat
    
    def __getitem__(self, index):
        """
        Get a data sample.
        
        Args:
            index: Sample index (may be repeated if repeat > 1)
        
        Returns:
            Tuple of (data_numpy, label, index)
        """
        # Handle repeat augmentation
        actual_index = index % len(self.data_list)
        
        data_numpy = self.data_list[actual_index].copy()
        label = self.label_list[actual_index]
        
        # Get valid frame number
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Apply temporal cropping/resizing
        data_numpy = tools.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        
        # Apply augmentations
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        
        if self.bone:
            # Convert to bone representation
            bone_data_numpy = np.zeros_like(data_numpy)
            # NTU bone pairs (from graph/ntu_rgb_d.py or feeders/bone_pairs.py)
            from feeders.bone_pairs import ntu_pairs
            for v1, v2 in ntu_pairs:
                # Convert to 0-indexed
                parent_idx = v1 - 1
                child_idx = v2 - 1
                if parent_idx >= 0 and parent_idx < 17:
                    bone_data_numpy[:, :, parent_idx, :] = data_numpy[:, :, parent_idx, :] - data_numpy[:, :, child_idx, :]
            data_numpy = bone_data_numpy
        
        if self.vel:
            # Convert to velocity representation
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        
        return data_numpy, label, index
    
    def top_k(self, score, top_k):
        """
        Calculate top-k accuracy.
        
        Args:
            score: Prediction scores (N, num_classes)
            top_k: List of k values for top-k accuracy
        
        Returns:
            Top-k accuracy value
        """
        rank = score.argsort()
        hit_top_k = [self.label_list[i] in rank[i, -top_k:] for i in range(len(self.label_list))]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
