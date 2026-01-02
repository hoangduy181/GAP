import numpy as np
import pickle

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 split_type='xsub', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, 
                 normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        Feeder for NTU60 2D skeleton data (17 joints, 2D coordinates)
        
        :param data_path: Path to pickle file (e.g., 'data/ntu2d/ntu60_2d.pkl')
        :param label_path: Not used (kept for compatibility)
        :param split: 'train' or 'test' (validation)
        :param split_type: 'xsub' (cross-subject) or 'xview' (cross-view)
        :param p_interval: Proportion of valid frames to use
        :param window_size: Target number of frames per sample (-1 for original length)
        :param random_rot: Apply random rotation (not recommended for 2D)
        :param normalization: Normalize coordinates
        :param debug: Use only first 100 samples
        :param bone: Convert to bone representation
        :param vel: Convert to velocity/motion representation
        """
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.split_type = split_type
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        # Ensure p_interval is a list (valid_crop_resize expects a list)
        if isinstance(p_interval, (int, float)):
            self.p_interval = [p_interval]
        else:
            self.p_interval = p_interval
        self.random_rot = random_rot  # Note: 2D rotation may not be meaningful
        self.bone = bone
        self.vel = vel
        
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        """Load data from pickle file and prepare for training/testing"""
        # Load pickle file
        with open(self.data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        split_data = data_dict['split']
        annotations = data_dict['annotations']
        
        # Create mapping from frame_dir to annotation index
        self.frame_dir_to_idx = {ann['frame_dir']: idx for idx, ann in enumerate(annotations)}
        
        # Get sample IDs for the requested split
        if self.split == 'train':
            split_key = f'{self.split_type}_train'
        elif self.split == 'test':
            split_key = f'{self.split_type}_val'  # Validation set
        else:
            raise ValueError(f"split must be 'train' or 'test', got '{self.split}'")
        
        if split_key not in split_data:
            raise ValueError(f"Split '{split_key}' not found in data. Available: {list(split_data.keys())}")
        
        sample_ids = split_data[split_key]
        
        # Filter annotations based on sample IDs
        self.indices = []
        self.data_list = []
        self.label_list = []
        self.sample_name = []
        
        for idx, sample_id in enumerate(sample_ids):
            if sample_id in self.frame_dir_to_idx:
                ann_idx = self.frame_dir_to_idx[sample_id]
                annotation = annotations[ann_idx]
                
                # Extract keypoint: shape (1, T, 17, 2)
                keypoint = annotation['keypoint']  # (1, T, 17, 2)
                
                # Convert to CTR-GCN format: (C, T, V, M) = (2, T, 17, 1)
                # Step 1: Remove M dimension -> (T, 17, 2)
                keypoint_reshaped = keypoint[0]  # (T, 17, 2)
                
                # Step 2: Transpose to (C, T, V) = (2, T, 17)
                keypoint_ctrgcn = keypoint_reshaped.transpose(2, 0, 1)  # (2, T, 17)
                
                # Step 3: Add M dimension -> (2, T, 17, 1)
                keypoint_final = keypoint_ctrgcn[:, :, :, np.newaxis]  # (2, T, 17, 1)
                
                self.data_list.append(keypoint_final)
                self.label_list.append(annotation['label'])
                self.sample_name.append(sample_id)
                self.indices.append(ann_idx)
        
        if self.debug:
            # Use only first 100 samples for debugging
            self.data_list = self.data_list[:100]
            self.label_list = self.label_list[:100]
            self.sample_name = self.sample_name[:100]
            self.indices = self.indices[:100]
        
        print(f"Loaded {len(self.data_list)} samples for {self.split} split ({self.split_type})")

    def get_mean_map(self):
        """Compute mean and std maps for normalization"""
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
        """Get a single sample"""
        # Get data: shape (C=2, T, V=17, M=1)
        data_numpy = np.array(self.data_list[index], dtype=np.float32)
        label = self.label_list[index]
        
        # Compute valid frames (non-zero frames)
        # For 2D data, check if any coordinate is non-zero
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Temporal cropping and resizing
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # Random rotation (if enabled) - Note: 2D rotation may not be meaningful
        if self.random_rot:
            # For 2D, rotation is around Z-axis only
            data_numpy = tools.random_rot(data_numpy)
        
        # Convert to bone representation if requested
        if self.bone:
            # For 17-joint COCO format, we need bone pairs
            # Using parent-child relationships from graph
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
        """Compute top-k accuracy"""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label_list)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
