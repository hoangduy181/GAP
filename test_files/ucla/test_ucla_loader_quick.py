#!/usr/bin/env python3
"""
Quick test script for UCLA YOLO Pose Loader

This is a minimal test that quickly verifies the loader works.
Run this first before running the full test suite.
"""

import os
import sys
from pathlib import Path

# Add project root to path (go up one level from test_files/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory for relative paths to work
os.chdir(project_root)

def quick_test():
    """Quick test of the loader"""
    print("=" * 70)
    print("Quick UCLA YOLO Loader Test")
    print("=" * 70)
    
    # Check if data.json exists
    data_json_path = "data/yolo_ucla/data.json"
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        print("\nPlease run preprocessing first:")
        print("  python main_multipart_yolo_ucla.py --mode process --config config/yolo_ucla/data_config.yaml")
        return False
    
    print(f"✅ Found data.json at {data_json_path}")
    
    # Try to import and initialize feeder
    try:
        from feeders.feeder_yolo_pose_ucla import Feeder
        
        print("\n📂 Initializing train feeder...")
        feeder = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="train",
            split="train",
            data_json_path=data_json_path,
            window_size=52,
            debug=True,  # Use debug mode for quick test
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        print(f"✅ Feeder initialized successfully")
        print(f"   - Number of samples: {len(feeder)}")
        
        if len(feeder) == 0:
            print("❌ ERROR: Feeder has no samples")
            return False
        
        # Try to get one sample
        print("\n📊 Testing sample retrieval...")
        data, label, index = feeder[0]
        
        print(f"✅ Sample retrieved successfully")
        print(f"   - Data shape: {data.shape}")
        print(f"   - Data dtype: {data.dtype}")
        print(f"   - Label: {label}")
        print(f"   - Index: {index}")
        if hasattr(feeder, 'sample_name') and len(feeder.sample_name) > index:
            print(f"   - Sample name: {feeder.sample_name[index]}")
        
        # Validate shape
        if data.shape != (2, 52, 17, 1):
            print(f"❌ ERROR: Expected shape (2, 52, 17, 1), got {data.shape}")
            return False
        
        print(f"\n🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
