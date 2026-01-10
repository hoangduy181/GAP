"""Joint ordering mapping between YOLO COCO format and joint17 format."""

import numpy as np


# YOLO COCO 17-joint format order (0-indexed)
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

# joint17.py format order (from graph/joint17.py, 0-indexed)
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

# Both formats use the same COCO 17-joint order, so no mapping is needed
# However, we provide this module for clarity and potential future changes

YOLO_COCO_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

JOINT17_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def yolo_to_joint17_keypoints(yolo_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert YOLO COCO keypoints to joint17 format.
    
    Since both formats use the same COCO 17-joint order, this is essentially
    a pass-through function, but ensures the correct shape and format.
    
    Args:
        yolo_keypoints: Keypoints from YOLO in shape (17, 3) or (T, 17, 3)
                       where last dimension is [x, y, confidence]
    
    Returns:
        Keypoints in joint17 format with same shape
    """
    # Both formats are identical, so just return the input
    # This function exists for API consistency and potential future changes
    return yolo_keypoints.copy()


def validate_keypoint_shape(keypoints: np.ndarray) -> bool:
    """
    Validate that keypoints have the correct shape.
    
    Args:
        keypoints: Keypoints array
    
    Returns:
        True if shape is valid
    """
    if keypoints.ndim == 2:
        # Single frame: (17, 3) or (17, 2)
        return keypoints.shape[0] == 17 and keypoints.shape[1] in [2, 3]
    elif keypoints.ndim == 3:
        # Multiple frames: (T, 17, 3) or (T, 17, 2)
        return keypoints.shape[1] == 17 and keypoints.shape[2] in [2, 3]
    else:
        return False
