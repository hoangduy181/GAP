"""Feeder for YOLO pose detection from video files for UCLA dataset."""

import numpy as np
import os
import glob
import pickle
import json
import re
from torch.utils.data import Dataset
from typing import Optional, List, Dict

from feeders import tools
from yolo_pose.yolo_pose_detector import YOLOPoseDetector


class Feeder(Dataset):
    """
    Feeder for YOLO pose detection from video files for UCLA dataset.
    
    Processes video files using YOLO pose detection and converts to CTR-GCN format.
    Extracts labels from video filenames (format: a{action}_s{subject}_e{episode}_v{view}.mp4)
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
                 window_size: int = 52,
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
                 data_json_path: Optional[str] = None):
        """
        Initialize YOLO Pose Feeder for UCLA dataset.
        
        Args:
            data_path: Path to directory containing video files
            label_path: 'train' or 'val' to determine split (or path to label file)
            split: 'train' or 'test' (for compatibility)
            p_interval: Proportion of valid frames to use
            window_size: Target number of frames per sample (default 52 for UCLA)
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
            data_json_path: Optional explicit path to data.json file
        """
        self.debug = debug
        self.data_json_path = data_json_path
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
        
        # Determine train/val split from label_path
        if label_path in ['train', 'val']:
            self.train_val = label_path
        elif label_path and 'val' in label_path.lower():
            self.train_val = 'val'
        else:
            self.train_val = 'train'
        
        # Ensure p_interval is a list
        if isinstance(p_interval, (int, float)):
            self.p_interval = [p_interval]
        else:
            self.p_interval = p_interval
        
        # Initialize YOLO detector
        self.yolo_detector = YOLOPoseDetector(
            model_path=yolo_model_path,
            conf_threshold=yolo_conf_threshold,
            device=yolo_device,
            tracking_strategy=yolo_tracking_strategy
        )
        
        # Load data
        self.load_data()
        
        if normalization:
            self.get_mean_map()
    
    def _extract_label_from_filename(self, filename: str) -> int:
        """
        Extract action label from UCLA filename format: a{action}_s{subject}_e{episode}_v{view}
        
        Args:
            filename: Video filename (e.g., 'a05_s04_e02_v03.mp4')
        
        Returns:
            Label (0-indexed, action - 1)
        """
        match = re.match(r'a(\d+)_', filename)
        if match:
            action = int(match.group(1))
            return action - 1  # Convert to 0-indexed
        return 0  # Default to 0 if pattern doesn't match
    
    def load_data(self):
        """Load data from preprocessed JSON files or video files."""
        # First, try to load from data.json if it exists
        data_json_path = None
        
        # Priority 1: Use explicitly provided data_json_path
        if self.data_json_path and os.path.exists(self.data_json_path):
            data_json_path = self.data_json_path
        # Priority 2: Check if data_path is a data.json file
        elif os.path.isfile(self.data_path) and self.data_path.endswith('data.json'):
            data_json_path = self.data_path
        # Priority 3: Check if data_path is a directory, look for data.json in it or parent
        elif os.path.isdir(self.data_path):
            # Check in data_path directory
            potential_data_json = os.path.join(self.data_path, 'data.json')
            if os.path.exists(potential_data_json):
                data_json_path = potential_data_json
            else:
                # Check in parent directory
                parent_dir = os.path.dirname(self.data_path)
                parent_data_json = os.path.join(parent_dir, 'data.json')
                if os.path.exists(parent_data_json):
                    data_json_path = parent_data_json
                else:
                    # Check in sibling directories (common pattern: data/yolo_ucla/data.json when data_path is data/yolo_ucla_json)
                    # Try common sibling directory names
                    sibling_dirs = ['yolo_ucla', 'ucla_yolo', 'data']
                    for sibling_dir in sibling_dirs:
                        sibling_data_json = os.path.join(parent_dir, sibling_dir, 'data.json')
                        if os.path.exists(sibling_data_json):
                            data_json_path = sibling_data_json
                            break
        
        # If data.json found, use it
        if data_json_path and os.path.exists(data_json_path):
            print(f"Loading from data.json: {data_json_path}")
            self._load_from_data_json(data_json_path)
            return
        
        # Fallback to old method: check if data_path points to JSON files (preprocessed data)
        if os.path.isdir(self.data_path):
            # Check if it contains JSON files (preprocessed) or video files
            json_files = glob.glob(os.path.join(self.data_path, '*.json'))
            # Filter out data.json files
            json_files = [f for f in json_files if not f.endswith('data.json') and not f.endswith('_data_dict.json')]
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(self.data_path, ext)))
                video_files.extend(glob.glob(os.path.join(self.data_path, ext.upper())))
            
            # Prefer JSON files if they exist (preprocessed data)
            if json_files:
                self._load_from_json_files(json_files)
                return
            elif video_files:
                self._load_from_video_files(video_files)
                return
            else:
                raise ValueError(f"No JSON or video files found in {self.data_path}")
        else:
            raise ValueError(f"data_path must be a directory or data.json file, got: {self.data_path}")
    
    def _load_from_data_json(self, data_json_path: str):
        """Load data from data.json file structure."""
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
        
        # Get the appropriate split
        split_data = data_json.get(self.train_val, [])
        if not split_data:
            raise ValueError(f"No {self.train_val} data found in data.json")
        
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, item in enumerate(split_data):
            json_path = item.get('json_path', '')
            full_file_name = item.get('full_file_name', '')
            file_name = item.get('file_name', full_file_name)
            label = item.get('label', 0)
            
            # Ensure label is 0-indexed
            if isinstance(label, int) and label > 0:
                label = label - 1
            
            # Check if JSON file exists
            if not os.path.exists(json_path):
                print(f"Warning: JSON file not found: {json_path}, skipping")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract skeletons: list of frames, each frame is list of 17 joints [x, y, conf]
                skeletons = data.get('skeletons', [])
                if not skeletons:
                    print(f"Warning: No skeletons found in {json_path}")
                    continue
                
                # Convert to numpy array: (T, 17, 3)
                keypoints_list = [np.array(frame) for frame in skeletons]
                
                # Convert to CTR-GCN format
                keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
                
                self.data_list.append(keypoints_array)
                self.label_list.append(label)
                self.sample_name.append(full_file_name or file_name)
                self.indices.append(idx)
                
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue
        
        if self.debug:
            # Debug mode: use only 1/5 of samples
            original_count = len(self.data_list)
            if original_count > 5:
                # Take every 5th sample (approximately 1/5)
                self.data_list = self.data_list[::5]
                self.label_list = self.label_list[::5]
                self.sample_name = self.sample_name[::5]
                self.indices = self.indices[::5]
                print(f"DEBUG MODE: Reduced dataset from {original_count} to {len(self.data_list)} samples (1/5)")
            else:
                # If already small, just use all
                print(f"DEBUG MODE: Using all {len(self.data_list)} samples (dataset already small)")
        
        print(f"Loaded {len(self.data_list)} samples from data.json for {self.train_val} split")
    
    def _load_from_json_files(self, json_files):
        """Load data from preprocessed JSON files."""
        json_files.sort()
        
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, json_path in enumerate(json_files):
            video_name = os.path.splitext(os.path.basename(json_path))[0]
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract skeletons: list of frames, each frame is list of 17 joints [x, y, conf]
                skeletons = data.get('skeletons', [])
                if not skeletons:
                    print(f"Warning: No skeletons found in {json_path}")
                    continue
                
                # Convert to numpy array: (T, 17, 3)
                keypoints_list = [np.array(frame) for frame in skeletons]
                
                # Extract label from filename or metadata
                label = data.get('metadata', {}).get('label', self._extract_label_from_filename(video_name))
                if isinstance(label, int) and label > 0:
                    label = label - 1  # Convert to 0-indexed
                else:
                    label = self._extract_label_from_filename(video_name)
                
                # Convert to CTR-GCN format
                keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
                
                self.data_list.append(keypoints_array)
                self.label_list.append(label)
                self.sample_name.append(video_name)
                self.indices.append(idx)
                
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue
        
        if self.debug:
            # Debug mode: use only 1/5 of samples
            original_count = len(self.data_list)
            if original_count > 5:
                # Take every 5th sample (approximately 1/5)
                self.data_list = self.data_list[::5]
                self.label_list = self.label_list[::5]
                self.sample_name = self.sample_name[::5]
                self.indices = self.indices[::5]
                print(f"DEBUG MODE: Reduced dataset from {original_count} to {len(self.data_list)} samples (1/5)")
            else:
                # If already small, just use all
                print(f"DEBUG MODE: Using all {len(self.data_list)} samples (dataset already small)")
        
        print(f"Loaded {len(self.data_list)} samples from JSON files for {self.train_val} split")
    
    def _load_from_video_files(self, video_files):
        """Load data from video files (fallback if JSON files not available)."""
        video_files.sort()
        
        # Process videos
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, video_path in enumerate(video_files):
            # Get label from filename
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label = self._extract_label_from_filename(video_name)
            
            # Check cache first
            cache_path = None
            if self.use_cache and self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_path = os.path.join(self.cache_dir, f"{video_name}.pkl")
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'rb') as f:
                            keypoints_list = pickle.load(f)
                        print(f"Loaded cached keypoints for {video_name}")
                    except Exception as e:
                        print(f"Failed to load cache for {video_name}: {e}, reprocessing...")
                        cache_path = None
            
            # Process video if not cached
            if cache_path is None or not os.path.exists(cache_path):
                try:
                    keypoints_list, metadata = self.yolo_detector.detect_video(video_path)
                    
                    # Save to cache
                    if self.use_cache and self.cache_dir:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(keypoints_list, f)
                        print(f"Cached keypoints for {video_name}")
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")
                    continue
            
            # Convert to CTR-GCN format
            # Note: YOLO outputs 17 joints, but UCLA uses 20 joints
            # We'll use 17 joints and pad to 20 if needed, or just use 17
            keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
            
            self.data_list.append(keypoints_array)
            self.label_list.append(label)
            self.sample_name.append(video_name)
            self.indices.append(idx)
        
        if self.debug:
            # Debug mode: use only 1/5 of samples
            original_count = len(self.data_list)
            if original_count > 5:
                # Take every 5th sample (approximately 1/5)
                self.data_list = self.data_list[::5]
                self.label_list = self.label_list[::5]
                self.sample_name = self.sample_name[::5]
                self.indices = self.indices[::5]
                print(f"DEBUG MODE: Reduced dataset from {original_count} to {len(self.data_list)} samples (1/5)")
            else:
                # If already small, just use all
                print(f"DEBUG MODE: Using all {len(self.data_list)} samples (dataset already small)")
        
        print(f"Loaded {len(self.data_list)} samples from video files for {self.train_val} split")
    
    def _yolo_to_ctrgcn_format(self, keypoints_list: List[np.ndarray]) -> np.ndarray:
        """
        Convert YOLO keypoints list to CTR-GCN format: (C=2, T, V=17, M=1)
        
        Note: YOLO outputs 17 COCO joints. UCLA uses 20 joints, but we'll use 17
        and let the model handle it (or pad to 20 if needed).
        
        Args:
            keypoints_list: List of keypoint arrays, each (17, 3) [x, y, confidence]
        
        Returns:
            Array in CTR-GCN format: (2, T, 17, 1)
        """
        if not keypoints_list:
            # Return zero array with single frame
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
        """Compute mean and std maps for normalization."""
        # Stack all data: (N, C, T, V, M)
        all_data = np.stack(self.data_list, axis=0)  # (N, 2, T, 17, 1)
        N, C, T, V, M = all_data.shape
        
        # Mean per channel and joint
        self.mean_map = all_data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        
        # Std per channel and joint
        self.std_map = all_data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))
    
    def __len__(self):
        return len(self.label_list) * self.repeat
    
    def __getitem__(self, index):
        """Get a single sample."""
        # Handle repeat for data augmentation
        actual_index = index % len(self.data_list)
        
        # Get data: shape (C=2, T, V=17, M=1)
        data_numpy = np.array(self.data_list[actual_index], dtype=np.float32)
        label = self.label_list[actual_index]
        
        # Compute valid frames (non-zero frames)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Temporal cropping and resizing
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # Random rotation (if enabled)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        
        # Convert to bone representation if requested
        if self.bone:
            # For 17-joint COCO format, we need bone pairs
            from graph.joint17 import inward_ori_index
            bone_data_numpy = np.zeros_like(data_numpy)
            for child, parent in inward_ori_index:
                # Convert to 0-indexed
                child_idx = child - 1
                parent_idx = parent - 1
                if parent_idx >= 0 and parent_idx < 17:
                    bone_data_numpy[:, :, child_idx, :] = data_numpy[:, :, child_idx, :] - data_numpy[:, :, parent_idx, :]
            data_numpy = bone_data_numpy
        
        # Convert to velocity/motion representation if requested
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        
        return data_numpy, label, index
    
    def top_k(self, score, top_k):
        """Compute top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label_list)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    """Import class by name string."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
