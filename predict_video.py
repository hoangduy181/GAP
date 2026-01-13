#!/usr/bin/env python3
"""
Predict action class for a single video using trained model weights.

Usage:
    python predict_video.py --video path/to/video.avi --weights path/to/model.pt --config config/yolo_ucla/data_config.yaml
"""

import os
import sys
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from feeders.feeder_yolo_pose_ucla import Feeder
from yolo_pose.yolo_pose_detector import YOLOPoseDetector
from tools import import_class


def predict_video(video_path: str, weights_path: str, config_path: str, 
                  yolo_model_path: str = 'yolo11n-pose.pt',
                  device: str = None):
    """
    Predict action class for a single video.
    
    Args:
        video_path: Path to video file
        weights_path: Path to trained model weights (.pt file)
        config_path: Path to config file
        yolo_model_path: Path to YOLO model weights
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        predicted_class: Predicted action class (0-9 for UCLA)
        confidence: Confidence scores for all classes
    """
    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    training_config = config_data.get('training', {})
    if not training_config:
        raise ValueError("No 'training' section found in config file")
    
    model_class = training_config.get('model')
    model_args = training_config.get('model_args', {})
    
    if not model_class:
        raise ValueError("'model' not specified in config")
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_device = int(device[0]) if device.startswith('cuda') else 0
    
    print(f"Using device: {device}")
    print(f"Loading model: {model_class}")
    
    # Load model
    Model = import_class(model_class)
    
    # Filter out 'head' from model_args (used by Processor, not Model)
    model_args_for_model = {k: v for k, v in model_args.items() if k != 'head'}
    model = Model(**model_args_for_model)
    model = model.cuda(output_device) if torch.cuda.is_available() else model
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    weights = torch.load(weights_path, map_location=f'cuda:{output_device}' if torch.cuda.is_available() else 'cpu')
    
    # Handle DataParallel weights (remove 'module.' prefix if present)
    if any(k.startswith('module.') for k in weights.keys()):
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
    
    model.load_state_dict(weights)
    model.eval()
    print("Model loaded successfully")
    
    # Process video with YOLO
    print(f"Processing video: {video_path}")
    detector = YOLOPoseDetector(
        model_path=yolo_model_path,
        conf_threshold=0.25,
        device=device,
        tracking_strategy='largest_bbox'
    )
    
    keypoints_list, metadata = detector.detect_video(video_path, return_frames=False)
    print(f"Extracted {len(keypoints_list)} frames")
    
    # Convert to CTR-GCN format using feeder's conversion method
    # Create a temporary feeder instance to use its conversion method
    feeder = Feeder(
        data_path="",  # Not used for single video
        label_path="train",
        split="train",
        window_size=52,
        debug=False,
        random_choose=False,
        repeat=1,
        yolo_model_path=yolo_model_path,
        use_cache=False
    )
    
    # Convert keypoints to CTR-GCN format: (2, T, 17, 1)
    keypoints_array = feeder._yolo_to_ctrgcn_format(keypoints_list)
    
    # Apply temporal cropping/resizing to match window_size
    from feeders import tools
    valid_frame_num = np.sum(keypoints_array.sum(0).sum(-1).sum(-1) != 0)
    keypoints_array = tools.valid_crop_resize(
        keypoints_array, 
        valid_frame_num, 
        [0.95],  # Use center crop for inference
        52  # window_size
    )
    
    # Add batch dimension: (1, 2, T, 17, 1)
    keypoints_tensor = torch.from_numpy(keypoints_array).float().unsqueeze(0)
    
    # Move to device
    if torch.cuda.is_available():
        keypoints_tensor = keypoints_tensor.cuda(output_device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            output, _, _, _ = model(keypoints_tensor)
    
    # Get predictions
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    all_confidences = probabilities[0].cpu().numpy()
    
    # Load class names if available
    class_names = None
    try:
        # Try to load from text file
        label_map_file = 'text/ucla_label_map.txt'
        if not os.path.exists(label_map_file):
            # Create a default mapping
            class_names = [f"Action {i+1}" for i in range(model_args.get('num_class', 10))]
        else:
            with open(label_map_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
    except:
        class_names = [f"Action {i+1}" for i in range(model_args.get('num_class', 10))]
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Predicted Class: {predicted_class} ({class_names[predicted_class] if class_names else 'N/A'})")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print("\nAll class probabilities:")
    for i, conf in enumerate(all_confidences):
        class_name = class_names[i] if class_names else f"Action {i+1}"
        marker = " <-- PREDICTED" if i == predicted_class else ""
        print(f"  Class {i}: {class_name:20s} - {conf:.4f} ({conf*100:.2f}%){marker}")
    print("="*60)
    
    return predicted_class, all_confidences


def main():
    parser = argparse.ArgumentParser(
        description='Predict action class for a single video using trained model')
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to video file')
    
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to trained model weights (.pt file)')
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., config/yolo_ucla/data_config.yaml)')
    
    parser.add_argument(
        '--yolo-model-path',
        type=str,
        default='yolo11n-pose.pt',
        help='Path to YOLO model weights (default: yolo11n-pose.pt)')
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cuda, cpu, or None for auto (default: None)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    predict_video(
        video_path=args.video,
        weights_path=args.weights,
        config_path=args.config,
        yolo_model_path=args.yolo_model_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
