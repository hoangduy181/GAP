"""Feeder for YOLO pose detection from video files."""

import numpy as np
import os
import glob
import pickle
from torch.utils.data import Dataset
from typing import Optional, List, Dict

from feeders import tools
from yolo_pose.yolo_pose_detector import YOLOPoseDetector


class Feeder(Dataset):
    """
    Feeder for YOLO pose detection from video files.
    
    Processes video files using YOLO pose detection and converts to CTR-GCN format.
    """
    
    def __init__(self, 
                 data_path: str,
                 label_path: Optional[str] = None,
                 p_interval: float = 1.0,
                 split: str = 'train',
                 split_type: str = 'xsub',
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
                 # YOLO-specific parameters
                 yolo_model_path: str = 'yolo11n-pose.pt',
                 yolo_conf_threshold: float = 0.25,
                 yolo_device: Optional[str] = None,
                 yolo_tracking_strategy: str = 'largest_bbox',
                 cache_dir: Optional[str] = None,
                 use_cache: bool = True):
        """
        Initialize YOLO Pose Feeder.
        
        Args:
            data_path: Path to directory containing video files or pickle cache file
            label_path: Path to label file (optional, can be inferred from video names)
            split: 'train' or 'test'
            split_type: Split type (for compatibility, not used for video files)
            p_interval: Proportion of valid frames to use
            window_size: Target number of frames per sample (-1 for original length)
            normalization: Normalize coordinates
            debug: Use only first 100 samples
            bone: Convert to bone representation
            vel: Convert to velocity/motion representation
            yolo_model_path: Path to YOLO model weights
            yolo_conf_threshold: Confidence threshold for YOLO detection
            yolo_device: Device for YOLO inference ('cuda', 'cpu', or None for auto)
            yolo_tracking_strategy: Strategy for person selection
            cache_dir: Directory to cache YOLO detections
            use_cache: Whether to use cached detections if available
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.split_type = split_type
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
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
    
    def load_data(self):
        """Load data from video files or cache."""
        # Check if data_path is a pickle cache file
        if self.data_path.endswith('.pkl') and os.path.exists(self.data_path):
            self._load_from_cache()
            return
        
        # Otherwise, treat as directory with video files
        if not os.path.isdir(self.data_path):
            raise ValueError(f"data_path must be a directory or pickle file, got: {self.data_path}")
        
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(self.data_path, ext)))
            video_files.extend(glob.glob(os.path.join(self.data_path, ext.upper())))
        
        if not video_files:
            raise ValueError(f"No video files found in {self.data_path}")
        
        video_files.sort()
        
        # Load labels if provided
        labels_dict = {}
        if self.label_path and os.path.exists(self.label_path):
            labels_dict = self._load_labels()
        
        # Process videos
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        self.indices = []
        
        for idx, video_path in enumerate(video_files):
            # Get label from filename or label file
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            label = labels_dict.get(video_name, 0)  # Default to 0 if no label
            
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
            keypoints_array = self._yolo_to_ctrgcn_format(keypoints_list)
            
            self.data_list.append(keypoints_array)
            self.label_list.append(label)
            self.sample_name.append(video_name)
            self.indices.append(idx)
        
        if self.debug:
            # Use only first 100 samples for debugging
            self.data_list = self.data_list[:100]
            self.label_list = self.label_list[:100]
            self.sample_name = self.sample_name[:100]
            self.indices = self.indices[:100]
        
        print(f"Loaded {len(self.data_list)} samples for {self.split} split")
    
    def _load_from_cache(self):
        """Load data from pickle cache file."""
        with open(self.data_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Expected format: dict with 'data', 'labels', 'sample_names'
        if isinstance(cache_data, dict):
            self.data_list = cache_data.get('data', [])
            self.label_list = cache_data.get('labels', [])
            self.sample_name = cache_data.get('sample_names', [])
            self.indices = list(range(len(self.data_list)))
        else:
            # Assume it's a list of data
            self.data_list = cache_data
            self.label_list = [0] * len(cache_data)
            self.sample_name = [f"sample_{i}" for i in range(len(cache_data))]
            self.indices = list(range(len(self.data_list)))
    
    def _load_labels(self) -> Dict[str, int]:
        """Load labels from label file."""
        labels_dict = {}
        if self.label_path and os.path.exists(self.label_path):
            # Try to load as text file (video_name: label)
            try:
                with open(self.label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ':' in line:
                            video_name, label = line.split(':', 1)
                            labels_dict[video_name.strip()] = int(label.strip())
            except Exception as e:
                print(f"Warning: Could not load labels from {self.label_path}: {e}")
        return labels_dict
    
    def _yolo_to_ctrgcn_format(self, keypoints_list: List[np.ndarray]) -> np.ndarray:
        """
        Convert YOLO keypoints list to CTR-GCN format: (C=2, T, V=17, M=1)
        
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
        return len(self.label_list)
    
    def __getitem__(self, index):
        """Get a single sample."""
        # Get data: shape (C=2, T, V=17, M=1)
        data_numpy = np.array(self.data_list[index], dtype=np.float32)
        label = self.label_list[index]
        
        # Compute valid frames (non-zero frames)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Temporal cropping and resizing
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # Random rotation (if enabled)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        
        # Convert to bone representation if requested
        if self.bone:
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
