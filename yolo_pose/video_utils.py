"""Video processing utilities for YOLO pose detection."""

import cv2
import os
from typing import Tuple, Optional


def extract_frames(video_path: str, max_frames: Optional[int] = None) -> list:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract (None for all frames)
    
    Returns:
        List of frames (numpy arrays)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames is not None and frame_count >= max_frames:
            break
    
    cap.release()
    return frames


def get_video_metadata(video_path: str) -> dict:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with video metadata (fps, frame_count, width, height)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height
    }


def sample_frames(frames: list, target_count: int) -> list:
    """
    Sample frames to match target count.
    
    Args:
        frames: List of frames
        target_count: Target number of frames
    
    Returns:
        List of sampled frames
    """
    if len(frames) == 0:
        return []
    
    if len(frames) <= target_count:
        return frames
    
    # Uniform sampling
    indices = [int(i * len(frames) / target_count) for i in range(target_count)]
    return [frames[i] for i in indices]
