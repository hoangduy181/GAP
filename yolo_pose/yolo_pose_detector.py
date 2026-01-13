"""YOLO Pose Detector for extracting 17-joint keypoints from videos."""

import numpy as np
from typing import List, Optional, Tuple, Dict, Union
from ultralytics import YOLO
import torch
from collections import defaultdict

from yolo_pose.video_utils import extract_frames, get_video_metadata
from yolo_pose.joint_mapping import yolo_to_joint17_keypoints, validate_keypoint_shape


class YOLOPoseDetector:
    """
    YOLO Pose Detector for extracting 17-joint keypoints from videos.
    
    Handles multi-person detection, tracking, and selection of the main person.
    """
    
    def __init__(self, 
                 model_path: str = 'yolo11n-pose.pt',
                 conf_threshold: float = 0.25,
                 device: Optional[str] = None,
                 tracking_strategy: str = 'largest_bbox'):
        """
        Initialize YOLO Pose Detector.
        
        Args:
            model_path: Path to YOLO model weights or model name (e.g., 'yolo11n-pose.pt')
            conf_threshold: Confidence threshold for pose detection
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            tracking_strategy: Strategy for selecting main person:
                              'largest_bbox' - person with largest bounding box
                              'highest_conf' - person with highest confidence
                              'track_id' - track person across frames
        """
        self.conf_threshold = conf_threshold
        self.tracking_strategy = tracking_strategy
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")
    
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect poses in a single frame.
        
        Args:
            frame: Input frame (numpy array, BGR format)
        
        Returns:
            List of detection dictionaries, each containing:
            - 'keypoints': (17, 3) array of [x, y, confidence]
            - 'bbox': bounding box [x1, y1, x2, y2]
            - 'conf': confidence score
            - 'track_id': tracking ID (if tracking enabled)
        """
        # save=False prevents YOLO from creating runs/pose/predict{n} output directories
        results = self.model(frame, conf=self.conf_threshold, verbose=False, save=False)
        
        detections = []
        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                for i, keypoint in enumerate(result.keypoints.data):
                    # keypoint shape: (17, 3) - [x, y, confidence]
                    keypoint_np = keypoint.cpu().numpy()
                    
                    # Get bounding box if available
                    bbox = None
                    conf = None
                    track_id = None
                    
                    if result.boxes is not None and len(result.boxes) > i:
                        box = result.boxes.data[i]
                        bbox = box[:4].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(box[4].cpu().numpy())  # confidence
                        
                        # Get track ID if tracking is enabled
                        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                            track_id = int(result.boxes.id[i].cpu().numpy())
                    
                    detections.append({
                        'keypoints': keypoint_np,
                        'bbox': bbox,
                        'conf': conf,
                        'track_id': track_id
                    })
        
        return detections
    
    def select_main_person(self, detections: List[Dict], 
                          previous_track_id: Optional[int] = None) -> Optional[Dict]:
        """
        Select the main person from multiple detections.
        
        Args:
            detections: List of detection dictionaries
            previous_track_id: Previous track ID to maintain consistency
        
        Returns:
            Selected detection dictionary or None if no detections
        """
        if not detections:
            return None
        
        if self.tracking_strategy == 'track_id' and previous_track_id is not None:
            # Try to find the same tracked person
            for det in detections:
                if det['track_id'] == previous_track_id:
                    return det
        
        # Select based on strategy
        if self.tracking_strategy == 'largest_bbox':
            # Select person with largest bounding box area
            best_det = None
            best_area = 0
            for det in detections:
                if det['bbox'] is not None:
                    x1, y1, x2, y2 = det['bbox']
                    area = (x2 - x1) * (y2 - y1)
                    if area > best_area:
                        best_area = area
                        best_det = det
            return best_det if best_det is not None else detections[0]
        
        elif self.tracking_strategy == 'highest_conf':
            # Select person with highest confidence
            best_det = max(detections, key=lambda x: x['conf'] if x['conf'] is not None else 0.0)
            return best_det
        
        else:
            # Default: return first detection
            return detections[0]
    
    def handle_occlusion(self, detections: List[Dict], 
                        previous_keypoints: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Handle cases where tracked person is occluded.
        
        Args:
            detections: List of current detections
            previous_keypoints: Previous frame's keypoints (17, 3)
        
        Returns:
            Keypoints array (17, 3) or None
        """
        if detections:
            return detections[0]['keypoints']
        
        # If no detections and we have previous keypoints, return them
        # (simple strategy - could be improved with interpolation)
        if previous_keypoints is not None:
            return previous_keypoints
        
        return None
    
    def detect_video(self, video_path: str, 
                    use_tracking: bool = True,
                    return_frames: bool = False) -> Union[Tuple[List[np.ndarray], Dict], Tuple[List[np.ndarray], Dict, List[np.ndarray]]]:
        """
        Process video file to extract keypoints for the main person.
        
        Args:
            video_path: Path to video file
            use_tracking: Whether to use tracking across frames
            return_frames: If True, also return the original frames for visualization
        
        Returns:
            Tuple of:
            - keypoints_list: List of keypoint arrays, each (17, 3) [x, y, confidence]
            - metadata: Dictionary with video metadata
            - frames: List of frames (only if return_frames=True, otherwise None)
        """
        # Get video metadata
        metadata = get_video_metadata(video_path)
        
        # Extract frames
        frames = extract_frames(video_path)
        
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        keypoints_list = []
        previous_track_id = None
        previous_keypoints = None
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Detect poses in frame
            detections = self.detect_frame(frame)
            
            # Select main person
            if detections:
                main_det = self.select_main_person(detections, previous_track_id)
                if main_det is not None:
                    keypoints = main_det['keypoints']
                    previous_keypoints = keypoints
                    
                    # Update track ID if available
                    if use_tracking and main_det['track_id'] is not None:
                        previous_track_id = main_det['track_id']
                else:
                    # Fallback to occlusion handling
                    keypoints = self.handle_occlusion(detections, previous_keypoints)
            else:
                # No detections - use occlusion handling
                keypoints = self.handle_occlusion([], previous_keypoints)
            
            if keypoints is not None:
                # Ensure keypoints are in correct format (17, 3)
                if keypoints.shape[0] != 17:
                    raise ValueError(f"Expected 17 keypoints, got {keypoints.shape[0]}")
                keypoints_list.append(keypoints)
            else:
                # Create zero keypoints if no detection
                keypoints_list.append(np.zeros((17, 3), dtype=np.float32))
        
        if return_frames:
            return keypoints_list, metadata, frames
        else:
            # Backward compatibility: return only 2 values when return_frames=False
            return keypoints_list, metadata
    
    def track_persons(self, video_path: str) -> List[np.ndarray]:
        """
        Track persons across frames using YOLO's built-in tracking.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of keypoint arrays for tracked person
        """
        # Use YOLO's tracking mode
        results = self.model.track(
            source=video_path,
            conf=self.conf_threshold,
            persist=True,
            verbose=False,
            save=False  # Disable saving output files to prevent runs/pose/predict{n} folders
        )
        
        keypoints_list = []
        main_track_id = None
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                # Get all detections with track IDs
                detections = []
                for i, keypoint in enumerate(result.keypoints.data):
                    keypoint_np = keypoint.cpu().numpy()
                    track_id = None
                    
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None and len(result.boxes.id) > i:
                        track_id = int(result.boxes.id[i].cpu().numpy())
                    
                    detections.append({
                        'keypoints': keypoint_np,
                        'track_id': track_id
                    })
                
                # Select main person (first time: largest, then track)
                if main_track_id is None:
                    # First frame: select largest
                    main_det = self.select_main_person(detections, None)
                    if main_det and main_det['track_id'] is not None:
                        main_track_id = main_det['track_id']
                    keypoints = main_det['keypoints'] if main_det else np.zeros((17, 3))
                else:
                    # Subsequent frames: track by ID
                    main_det = None
                    for det in detections:
                        if det['track_id'] == main_track_id:
                            main_det = det
                            break
                    
                    if main_det:
                        keypoints = main_det['keypoints']
                    else:
                        # Lost track - use previous or zero
                        keypoints = keypoints_list[-1] if keypoints_list else np.zeros((17, 3))
                
                keypoints_list.append(keypoints)
            else:
                # No detections - use previous or zero
                keypoints = keypoints_list[-1] if keypoints_list else np.zeros((17, 3))
                keypoints_list.append(keypoints)
        
        return keypoints_list
