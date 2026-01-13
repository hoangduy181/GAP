#!/usr/bin/env python3
"""
Test script for NTU60 JSON file loading

Tests loading skeleton data from preprocessed JSON files.
"""

import os
import sys
import json
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def test_json_loading():
    """Test loading JSON files directly"""
    print("=" * 70)
    print("NTU60 JSON File Loading Test")
    print("=" * 70)
    
    # Check if JSON files exist
    json_dir = "data/ntu60_json/train"
    if not os.path.exists(json_dir):
        print(f"❌ ERROR: JSON directory not found: {json_dir}")
        print("\nPlease run preprocessing first:")
        print("  python main_multipart_yolo.py --mode preprocess --config config/ntu60_yolo.yaml")
        return False
    
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    json_files = [f for f in json_files if not f.endswith('_data_dict.json')]
    
    if not json_files:
        print(f"❌ ERROR: No JSON files found in {json_dir}")
        return False
    
    print(f"✅ Found {len(json_files)} JSON files in {json_dir}")
    
    # Test loading a few JSON files
    print(f"\n📊 Testing JSON file structure...")
    test_files = json_files[:5]  # Test first 5 files
    
    for json_file in test_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            if 'skeletons' not in data:
                print(f"   ❌ ERROR: Missing 'skeletons' field in {os.path.basename(json_file)}")
                return False
            
            if 'metadata' not in data:
                print(f"   ❌ ERROR: Missing 'metadata' field in {os.path.basename(json_file)}")
                return False
            
            skeletons = data['skeletons']
            metadata = data['metadata']
            
            # Validate structure
            if not isinstance(skeletons, list):
                print(f"   ❌ ERROR: 'skeletons' should be a list in {os.path.basename(json_file)}")
                return False
            
            if len(skeletons) == 0:
                print(f"   ⚠️  WARNING: No skeletons in {os.path.basename(json_file)}")
                continue
            
            # Check first skeleton frame
            first_frame = skeletons[0]
            if not isinstance(first_frame, list):
                print(f"   ❌ ERROR: Skeleton frame should be a list in {os.path.basename(json_file)}")
                return False
            
            if len(first_frame) != 17:
                print(f"   ⚠️  WARNING: Expected 17 joints, got {len(first_frame)} in {os.path.basename(json_file)}")
            
            # Check joint format (should be [x, y, confidence])
            if len(first_frame[0]) != 3:
                print(f"   ⚠️  WARNING: Expected [x, y, conf] format, got {len(first_frame[0])} values")
            
            # Check metadata
            required_metadata = ['label_action', 'video_name', 'num_frames']
            for key in required_metadata:
                if key not in metadata:
                    print(f"   ⚠️  WARNING: Missing metadata key '{key}' in {os.path.basename(json_file)}")
            
            print(f"   ✅ {os.path.basename(json_file)}: {len(skeletons)} frames, label={metadata.get('label_action', 'N/A')}")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ ERROR: Invalid JSON in {os.path.basename(json_file)}: {str(e)}")
            return False
        except Exception as e:
            print(f"   ❌ ERROR: Failed to process {os.path.basename(json_file)}: {str(e)}")
            return False
    
    print(f"\n✅ JSON file structure validation passed!")
    
    # Test loading with feeder
    print(f"\n📊 Testing feeder loading from JSON files...")
    try:
        from feeders.feeder_yolo_pose_ntu60 import Feeder
        
        feeder = Feeder(
            data_path=json_dir,
            split="train",
            window_size=64,
            debug=True,
            random_choose=False,
            repeat=1,
            use_cache=False
        )
        
        print(f"✅ Feeder loaded {len(feeder)} samples from JSON files")
        
        if len(feeder) > 0:
            data, label, index = feeder[0]
            print(f"   - Sample shape: {data.shape}")
            print(f"   - Sample label: {label}")
            print(f"   ✅ Feeder can successfully load from JSON files")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Feeder loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_json_loading()
    sys.exit(0 if success else 1)
