#!/usr/bin/env python3
"""
Test script for eval-rgb mode (RGB-based evaluation).

This script tests the evaluation functionality on videos using RGB frames.
Note: Currently eval-rgb uses the same skeleton extraction as eval-ske.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import subprocess
import tempfile


def test_eval_rgb_single_video():
    """Test eval-rgb mode with a single video file."""
    print("="*60)
    print("Test: eval-rgb mode with single video")
    print("="*60)
    
    # Check if required files exist
    config_path = "config/yolo_ucla/data_config.yaml"
    if not os.path.exists(config_path):
        print(f"SKIP: Config file not found: {config_path}")
        return False
    
    # Try to find a model weight file
    work_dir = "work_dir/yolo_ucla/ctrgcn_joint"
    if not os.path.exists(work_dir):
        print(f"SKIP: Work directory not found: {work_dir}")
        return False
    
    import glob
    weight_files = glob.glob(os.path.join(work_dir, "runs-*.pt"))
    if not weight_files:
        print(f"SKIP: No model weights found in {work_dir}")
        return False
    
    weights_path = weight_files[0]
    print(f"Using weights: {weights_path}")
    
    # Try to find a test video
    video_dir = "data/multiview_action_videos"
    if not os.path.exists(video_dir):
        print(f"SKIP: Video directory not found: {video_dir}")
        return False
    
    # Find first video file
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov')):
                video_files.append(os.path.join(root, file))
                break
        if video_files:
            break
    
    if not video_files:
        print(f"SKIP: No video files found in {video_dir}")
        return False
    
    video_path = video_files[0]
    print(f"Using test video: {video_path}")
    
    # Run eval-rgb
    cmd = [
        sys.executable,
        "main_multipart_yolo_ucla.py",
        "--mode", "eval-rgb",
        "--config", config_path,
        "--weights", weights_path,
        "--video", video_path
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Test passed: eval-rgb mode executed successfully")
            # Check that it mentions skeleton extraction
            if "skeleton" in result.stdout.lower() or "keypoint" in result.stdout.lower():
                print("  ✓ Confirmed: Uses skeleton extraction (as expected)")
            return True
        else:
            print(f"\n✗ Test failed: Command returned exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Test failed: Command timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        return False


def test_eval_rgb_video_list():
    """Test eval-rgb mode with a video list file."""
    print("\n" + "="*60)
    print("Test: eval-rgb mode with video list")
    print("="*60)
    
    # Check if required files exist
    config_path = "config/yolo_ucla/data_config.yaml"
    if not os.path.exists(config_path):
        print(f"SKIP: Config file not found: {config_path}")
        return False
    
    # Try to find a model weight file
    work_dir = "work_dir/yolo_ucla/ctrgcn_joint"
    if not os.path.exists(work_dir):
        print(f"SKIP: Work directory not found: {work_dir}")
        return False
    
    import glob
    weight_files = glob.glob(os.path.join(work_dir, "runs-*.pt"))
    if not weight_files:
        print(f"SKIP: No model weights found in {work_dir}")
        return False
    
    weights_path = weight_files[0]
    print(f"Using weights: {weights_path}")
    
    # Try to find test videos
    video_dir = "data/multiview_action_videos"
    if not os.path.exists(video_dir):
        print(f"SKIP: Video directory not found: {video_dir}")
        return False
    
    # Find first 2 video files
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.avi', '.mp4', '.mov')):
                video_files.append(os.path.join(root, file))
                if len(video_files) >= 2:
                    break
        if len(video_files) >= 2:
            break
    
    if len(video_files) < 2:
        print(f"SKIP: Need at least 2 video files, found {len(video_files)}")
        return False
    
    # Create temporary video list file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for vid_path in video_files:
            f.write(f"{vid_path}\n")
        video_list_path = f.name
    
    print(f"Created video list: {video_list_path}")
    print(f"  Videos: {len(video_files)}")
    
    try:
        # Run eval-rgb
        cmd = [
            sys.executable,
            "main_multipart_yolo_ucla.py",
            "--mode", "eval-rgb",
            "--config", config_path,
            "--weights", weights_path,
            "--video-list", video_list_path
        ]
        
        print(f"\nRunning command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✓ Test passed: eval-rgb mode with video list executed successfully")
            return True
        else:
            print(f"\n✗ Test failed: Command returned exit code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ Test failed: Command timed out after 600 seconds")
        return False
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(video_list_path):
            os.unlink(video_list_path)


def main():
    """Run all tests."""
    print("="*60)
    print("Testing eval-rgb mode")
    print("="*60)
    
    results = []
    
    # Test 1: Single video
    results.append(("Single Video", test_eval_rgb_single_video()))
    
    # Test 2: Video list
    results.append(("Video List", test_eval_rgb_video_list()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
