"""
Visualization utilities for skeleton keypoints.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# COCO 17-joint skeleton connections
# Format: (parent_joint_idx, child_joint_idx)
COCO_SKELETON = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (5, 6),   # left_shoulder -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip -> right_hip
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle
]

# Joint names for COCO 17 format
COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def draw_skeleton_on_frame(frame: np.ndarray, 
                           keypoints: np.ndarray,
                           conf_threshold: float = 0.3,
                           joint_radius: int = 4,
                           line_thickness: int = 2,
                           show_joint_names: bool = False) -> np.ndarray:
    """
    Draw skeleton keypoints on a frame.
    
    Args:
        frame: Input frame (BGR format)
        keypoints: Keypoints array (17, 3) [x, y, confidence]
        conf_threshold: Minimum confidence to draw a joint
        joint_radius: Radius of joint circles
        line_thickness: Thickness of skeleton lines
        show_joint_names: Whether to show joint names
    
    Returns:
        Frame with skeleton drawn (BGR format)
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Colors: BGR format
    joint_color = (0, 255, 0)  # Green for joints
    skeleton_color = (0, 255, 255)  # Yellow for skeleton lines
    text_color = (255, 255, 255)  # White for text
    
    # Draw skeleton connections first (so joints appear on top)
    for parent_idx, child_idx in COCO_SKELETON:
        if parent_idx >= len(keypoints) or child_idx >= len(keypoints):
            continue
        
        parent_kp = keypoints[parent_idx]
        child_kp = keypoints[child_idx]
        
        # Check confidence
        if len(parent_kp) >= 3 and parent_kp[2] < conf_threshold:
            continue
        if len(child_kp) >= 3 and child_kp[2] < conf_threshold:
            continue
        
        # Get coordinates
        px, py = int(parent_kp[0]), int(parent_kp[1])
        cx, cy = int(child_kp[0]), int(child_kp[1])
        
        # Check if coordinates are valid
        if 0 <= px < w and 0 <= py < h and 0 <= cx < w and 0 <= cy < h:
            cv2.line(frame, (px, py), (cx, cy), skeleton_color, line_thickness)
    
    # Draw joints
    for i, kp in enumerate(keypoints):
        if len(kp) >= 3 and kp[2] < conf_threshold:
            continue
        
        x, y = int(kp[0]), int(kp[1])
        
        # Check if coordinates are valid
        if 0 <= x < w and 0 <= y < h:
            # Draw joint circle
            cv2.circle(frame, (x, y), joint_radius, joint_color, -1)
            
            # Draw joint index
            cv2.putText(frame, str(i), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
            # Draw joint name if requested
            if show_joint_names and i < len(COCO_JOINT_NAMES):
                cv2.putText(frame, COCO_JOINT_NAMES[i], (x + 5, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
    
    return frame


def visualize_keypoints_on_video(video_path: str,
                                 keypoints_list: List[np.ndarray],
                                 output_path: str,
                                 fps: Optional[float] = None,
                                 conf_threshold: float = 0.3) -> None:
    """
    Create a video with skeleton keypoints overlaid.
    
    Args:
        video_path: Path to input video
        keypoints_list: List of keypoint arrays, each (17, 3)
        output_path: Path to save output video
        fps: FPS for output video (if None, uses input video FPS)
        conf_threshold: Minimum confidence to draw joints
    """
    from yolo_pose.video_utils import get_video_metadata, extract_frames
    
    # Get video metadata
    metadata = get_video_metadata(video_path)
    if fps is None:
        fps = metadata.get('fps', 30.0)
    
    # Extract frames
    frames = extract_frames(video_path)
    
    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    # Get video dimensions
    h, w = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Process each frame
    num_frames = min(len(frames), len(keypoints_list))
    for i in range(num_frames):
        frame = frames[i]
        keypoints = keypoints_list[i]
        
        # Draw skeleton on frame
        frame_with_skeleton = draw_skeleton_on_frame(
            frame, keypoints, conf_threshold=conf_threshold
        )
        
        # Add frame number and info
        cv2.putText(frame_with_skeleton, f"Frame: {i+1}/{num_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Count visible joints
        visible_joints = np.sum(keypoints[:, 2] >= conf_threshold) if len(keypoints[0]) >= 3 else 17
        cv2.putText(frame_with_skeleton, f"Visible joints: {visible_joints}/17",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame_with_skeleton)
    
    out.release()
    print(f"Visualization video saved to: {output_path}")


def save_keypoint_frames(video_path: str,
                        keypoints_list: List[np.ndarray],
                        output_dir: str,
                        sample_interval: int = 10,
                        conf_threshold: float = 0.3) -> None:
    """
    Save sample frames with skeleton keypoints overlaid.
    
    Args:
        video_path: Path to input video
        keypoints_list: List of keypoint arrays, each (17, 3)
        output_dir: Directory to save frame images
        sample_interval: Save every Nth frame (e.g., 10 = every 10th frame)
        conf_threshold: Minimum confidence to draw joints
    """
    import os
    from yolo_pose.video_utils import extract_frames
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames
    frames = extract_frames(video_path)
    
    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Save sample frames
    saved_count = 0
    num_frames = min(len(frames), len(keypoints_list))
    
    for i in range(0, num_frames, sample_interval):
        frame = frames[i]
        keypoints = keypoints_list[i]
        
        # Draw skeleton on frame
        frame_with_skeleton = draw_skeleton_on_frame(
            frame, keypoints, conf_threshold=conf_threshold
        )
        
        # Add frame number
        cv2.putText(frame_with_skeleton, f"Frame {i+1}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame
        output_file = os.path.join(output_dir, f"{video_name}_frame_{i+1:04d}.jpg")
        cv2.imwrite(output_file, frame_with_skeleton)
        saved_count += 1
    
    print(f"Saved {saved_count} sample frames to {output_dir}")
