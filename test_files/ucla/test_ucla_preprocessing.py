#!/usr/bin/env python3
"""
Test script for UCLA YOLO preprocessing step

This script tests the preprocessing functionality:
1. Checks if video files exist
2. Tests YOLO pose detection on a single video
3. Validates output JSON format
4. Tests data.json generation
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path (go up one level from test_files/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory for relative paths to work
os.chdir(project_root)


def test_video_files_exist():
    """Test if video files exist in the expected location"""
    print("=" * 70)
    print("TEST 1: Video Files Check")
    print("=" * 70)
    
    video_dir = "data/multiview_action_videos"
    
    if not os.path.exists(video_dir):
        print(f"❌ ERROR: Video directory not found at {video_dir}")
        return False
    
    print(f"✅ Video directory exists: {video_dir}")
    
    # Check for action class folders
    action_folders = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d)) and d.startswith('a')]
    action_folders.sort()
    
    print(f"\n📂 Found {len(action_folders)} action class folders:")
    for folder in action_folders[:10]:  # Show first 10
        folder_path = os.path.join(video_dir, folder)
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4', '.mov'))]
        print(f"   - {folder}: {len(video_files)} videos")
    
    if len(action_folders) > 10:
        print(f"   ... and {len(action_folders) - 10} more")
    
    # Count total videos
    total_videos = 0
    for folder in action_folders:
        folder_path = os.path.join(video_dir, folder)
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4', '.mov'))]
        total_videos += len(video_files)
    
    print(f"\n📊 Total videos found: {total_videos}")
    
    if total_videos == 0:
        print("❌ ERROR: No video files found")
        return False
    
    return True


def test_yolo_detection():
    """Test YOLO pose detection on a single video"""
    print("\n" + "=" * 70)
    print("TEST 2: YOLO Pose Detection (Single Video)")
    print("=" * 70)
    
    video_dir = "data/multiview_action_videos"
    
    # Find first video file
    video_file = None
    for folder in sorted(os.listdir(video_dir)):
        folder_path = os.path.join(video_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        for file in os.listdir(folder_path):
            if file.endswith(('.avi', '.mp4', '.mov')):
                video_file = os.path.join(folder_path, file)
                break
        
        if video_file:
            break
    
    if not video_file:
        print("❌ ERROR: No video file found for testing")
        return False
    
    print(f"📹 Testing with video: {os.path.basename(video_file)}")
    
    try:
        from yolo_pose.yolo_pose_detector import YOLOPoseDetector
        
        print("\n🔧 Initializing YOLO detector...")
        detector = YOLOPoseDetector(
            model_path="yolo11n-pose.pt",
            conf_threshold=0.25,
            device=None,  # Auto
            tracking_strategy="largest_bbox"
        )
        print("✅ YOLO detector initialized")
        
        print("\n🔍 Running pose detection...")
        keypoints_list, metadata = detector.detect_video(video_file) 
        
        if keypoints_list is None or len(keypoints_list) == 0:
            print("❌ ERROR: No detections returned")
            return False
        
        print(f"✅ Detected poses in {len(keypoints_list)} frames")
        print(f"   Metadata: {metadata}")
        print(f"   First 3 keypoints shapes: {[kp.shape for kp in keypoints_list[:3]]}")
        
        # Validate results format
        for i, keypoints in enumerate(keypoints_list[:3]):  # Check first 3
            # keypoints should be a numpy array with shape (17, 3) - [x, y, confidence]
            if not isinstance(keypoints, np.ndarray):
                print(f"❌ ERROR: Keypoints should be numpy array, got {type(keypoints)}")
                return False
            
            if keypoints.shape != (17, 3):  # 17 joints, (x, y, confidence)
                print(f"❌ ERROR: Invalid keypoints shape: {keypoints.shape}, expected (17, 3)")
                return False
            
            # Check if keypoints are valid (not all zeros)
            if np.all(keypoints == 0):
                print(f"   ⚠️  Frame {i}: All keypoints are zero (no detection)")
            else:
                # Count non-zero keypoints
                non_zero = np.sum(np.any(keypoints[:, :2] != 0, axis=1))  # Check x, y (ignore confidence)
                print(f"   ✅ Frame {i}: {non_zero}/17 keypoints detected (shape: {keypoints.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: YOLO detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_json_output_format():
    """Test if preprocessed JSON files have correct format"""
    print("\n" + "=" * 70)
    print("TEST 3: Preprocessed JSON Format Validation")
    print("=" * 70)
    
    output_dir = "data/yolo_ucla_json/train"
    
    if not os.path.exists(output_dir):
        print(f"⚠️  WARNING: Output directory not found at {output_dir}")
        print("   Run preprocessing first to generate JSON files")
        return False
    
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if len(json_files) == 0:
        print(f"⚠️  WARNING: No JSON files found in {output_dir}")
        return False
    
    # Filter out metadata/index files (these don't have skeleton data)
    metadata_files = ['train_data_dict.json', 'val_data_dict.json', 'data.json', '_data_dict.json']
    skeleton_files = [f for f in json_files if not any(f.endswith(mf) or mf in f for mf in metadata_files)]
    
    print(f"📂 Found {len(json_files)} JSON files")
    print(f"   - Metadata files: {len(json_files) - len(skeleton_files)}")
    print(f"   - Skeleton data files: {len(skeleton_files)}")
    
    if len(skeleton_files) == 0:
        print(f"⚠️  WARNING: No skeleton data files found (only metadata files)")
        return False
    
    # Test a few files
    print(f"\n📊 Validating JSON format...")
    tested = 0
    for json_file in skeleton_files[:5]:  # Test first 5 skeleton files
        json_path = os.path.join(output_dir, json_file)
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check required fields - actual format uses "skeletons" (plural) and label in metadata
            if 'skeletons' not in data:
                print(f"   ❌ {json_file}: Missing 'skeletons' field")
                return False
            
            # Check skeletons format
            skeletons = data['skeletons']
            if not isinstance(skeletons, list):
                print(f"   ❌ {json_file}: skeletons should be list, got {type(skeletons)}")
                return False
            
            if len(skeletons) == 0:
                print(f"   ⚠️  {json_file}: Empty skeletons")
                continue
            
            # Check first frame - should be a list of 17 joints, each joint is [x, y, confidence]
            first_frame = skeletons[0]
            if not isinstance(first_frame, list):
                print(f"   ❌ {json_file}: Frame should be list, got {type(first_frame)}")
                return False
            
            # Expected format: skeletons is list of frames, each frame is list of 17 joints [x, y, conf]
            if len(first_frame) != 17:
                print(f"   ⚠️  {json_file}: Expected 17 joints per frame, got {len(first_frame)}")
            
            # Check first joint format
            if len(first_frame) > 0:
                first_joint = first_frame[0]
                if not isinstance(first_joint, list) or len(first_joint) != 3:
                    print(f"   ⚠️  {json_file}: Joint should be [x, y, conf], got {type(first_joint)}")
            
            # Check metadata (optional but usually present)
            metadata = data.get('metadata', {})
            label = metadata.get('label', None)
            
            print(f"   ✅ {json_file}: Valid format ({len(skeletons)} frames, label={label})")
            tested += 1
            
        except json.JSONDecodeError as e:
            print(f"   ❌ {json_file}: Invalid JSON: {str(e)}")
            return False
        except Exception as e:
            print(f"   ❌ {json_file}: Error: {str(e)}")
            return False
    
    print(f"\n✅ Validated {tested} JSON files")
    return True


def test_data_json_structure():
    """Test data.json structure"""
    print("\n" + "=" * 70)
    print("TEST 4: data.json Structure Validation")
    print("=" * 70)
    
    data_json_path = "data/yolo_ucla/data.json"
    
    if not os.path.exists(data_json_path):
        print(f"⚠️  WARNING: data.json not found at {data_json_path}")
        print("   Run preprocessing to generate data.json")
        return False
    
    try:
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
        
        # Check structure
        if not isinstance(data_json, dict):
            print(f"❌ ERROR: data.json should be a dict, got {type(data_json)}")
            return False
        
        required_keys = ['train', 'val']
        missing = [key for key in required_keys if key not in data_json]
        
        if missing:
            print(f"❌ ERROR: Missing keys in data.json: {missing}")
            return False
        
        print(f"✅ data.json has required keys: {list(data_json.keys())}")
        
        # Check train split
        train_data = data_json['train']
        if not isinstance(train_data, list):
            print(f"❌ ERROR: 'train' should be a list, got {type(train_data)}")
            return False
        
        print(f"   - Train samples: {len(train_data)}")
        
        # Check val split
        val_data = data_json['val']
        if not isinstance(val_data, list):
            print(f"❌ ERROR: 'val' should be a list, got {type(val_data)}")
            return False
        
        print(f"   - Val samples: {len(val_data)}")
        
        # Check sample structure
        if len(train_data) > 0:
            sample = train_data[0]
            required_sample_keys = ['json_path', 'label', 'full_file_name']
            missing_keys = [key for key in required_sample_keys if key not in sample]
            
            if missing_keys:
                print(f"❌ ERROR: Sample missing keys: {missing_keys}")
                return False
            
            print(f"   ✅ Sample structure valid")
            print(f"      Example: {sample.get('full_file_name', 'N/A')} -> label {sample.get('label', 'N/A')}")
        
        # Check label range
        train_labels = [item['label'] for item in train_data]
        val_labels = [item['label'] for item in val_data]
        all_labels = set(train_labels + val_labels)
        
        print(f"\n📊 Label statistics:")
        print(f"   - Unique labels: {sorted(all_labels)}")
        print(f"   - Label range: {min(all_labels)} to {max(all_labels)}")
        
        if max(all_labels) >= 10:
            print(f"   ⚠️  WARNING: Labels exceed expected range (0-9)")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all preprocessing tests"""
    print("\n" + "=" * 70)
    print("UCLA YOLO Preprocessing Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        ("Video Files Check", test_video_files_exist),
        ("YOLO Detection", test_yolo_detection),
        ("JSON Output Format", test_json_output_format),
        ("data.json Structure", test_data_json_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
