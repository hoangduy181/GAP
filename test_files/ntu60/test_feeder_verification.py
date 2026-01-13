#!/usr/bin/env python3
"""
Quick verification script for NTU60 YOLO Feeder
Tests if the feeder can load data correctly
"""

import os
import sys
from pathlib import Path

# Add project root to path (go up one level from test_files/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory for relative paths to work
os.chdir(project_root)

def verify_feeder():
    """Verify the feeder works correctly"""
    print("=" * 70)
    print("NTU60 YOLO Feeder Verification")
    print("=" * 70)
    
    # Check prerequisites
    data_json_path = "data/ntu60_splits/data.json"
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        print("\nPlease run preprocessing first:")
        print("  python main_multipart_yolo.py --mode preprocess --config config/ntu60_yolo.yaml")
        return False
    
    print(f"✅ Found data.json at {data_json_path}")
    
    try:
        from feeders.feeder_yolo_pose_ntu60 import Feeder
        import numpy as np
        
        # Test train feeder
        print("\n" + "-" * 70)
        print("Testing TRAIN feeder...")
        print("-" * 70)
        
        train_feeder = Feeder(
            data_path="data/ntu60_json",
            split="train",
            data_json_path=data_json_path,
            window_size=64,
            debug=True,  # Use debug mode for quick test
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        print(f"✅ Train feeder initialized")
        print(f"   - Number of samples: {len(train_feeder)}")
        
        if len(train_feeder) == 0:
            print("❌ ERROR: Train feeder has no samples")
            return False
        
        # Test sample retrieval
        print(f"\n📊 Testing sample retrieval from train feeder...")
        data, label, index = train_feeder[0]
        
        print(f"   ✅ Sample retrieved:")
        print(f"      - Data shape: {data.shape}")
        print(f"      - Data dtype: {data.dtype}")
        print(f"      - Label: {label} (type: {type(label).__name__})")
        print(f"      - Index: {index}")
        if hasattr(train_feeder, 'sample_name') and len(train_feeder.sample_name) > index:
            print(f"      - Sample name: {train_feeder.sample_name[index]}")
        
        # Validate shape (C=2, T=64, V=17, M=1)
        expected_shape = (2, 64, 17, 1)
        if data.shape != expected_shape:
            print(f"   ❌ ERROR: Expected shape {expected_shape}, got {data.shape}")
            return False
        
        # Validate data range (should be reasonable coordinates)
        if np.any(np.isnan(data)):
            print(f"   ⚠️  WARNING: Data contains NaN values")
        if np.any(np.isinf(data)):
            print(f"   ⚠️  WARNING: Data contains Inf values")
        
        # Test val feeder
        print("\n" + "-" * 70)
        print("Testing VAL feeder...")
        print("-" * 70)
        
        val_feeder = Feeder(
            data_path="data/ntu60_json",
            split="val",
            data_json_path=data_json_path,
            window_size=64,
            debug=True,  # Use debug mode for quick test
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        print(f"✅ Val feeder initialized")
        print(f"   - Number of samples: {len(val_feeder)}")
        
        if len(val_feeder) == 0:
            print("❌ ERROR: Val feeder has no samples")
            return False
        
        # Test sample retrieval
        print(f"\n📊 Testing sample retrieval from val feeder...")
        data, label, index = val_feeder[0]
        
        print(f"   ✅ Sample retrieved:")
        print(f"      - Data shape: {data.shape}")
        print(f"      - Data dtype: {data.dtype}")
        print(f"      - Label: {label}")
        print(f"      - Index: {index}")
        if hasattr(val_feeder, 'sample_name') and len(val_feeder.sample_name) > index:
            print(f"      - Sample name: {val_feeder.sample_name[index]}")
        
        # Validate shape
        if data.shape != expected_shape:
            print(f"   ❌ ERROR: Expected shape {expected_shape}, got {data.shape}")
            return False
        
        # Test multiple samples
        print(f"\n📊 Testing multiple samples (first 3)...")
        for i in range(min(3, len(train_feeder))):
            try:
                data, label, index = train_feeder[i]
                sample_name = train_feeder.sample_name[index] if hasattr(train_feeder, 'sample_name') and len(train_feeder.sample_name) > index else "N/A"
                print(f"   ✅ Sample {i}: shape={data.shape}, label={label}, name={sample_name[:40]}...")
            except Exception as e:
                print(f"   ❌ ERROR retrieving sample {i}: {str(e)}")
                return False
        
        # Test label distribution
        print(f"\n📊 Label distribution in train set:")
        train_labels = train_feeder.label_list
        unique_labels = sorted(set(train_labels))
        print(f"   - Unique labels: {len(unique_labels)} unique labels")
        print(f"   - Label range: {min(train_labels)} to {max(train_labels)}")
        print(f"   - Total samples: {len(train_labels)}")
        
        # Check if labels are in expected range (0-59 for NTU60)
        if max(train_labels) >= 60:
            print(f"   ⚠️  WARNING: Labels exceed expected range (0-59)")
        if min(train_labels) < 0:
            print(f"   ⚠️  WARNING: Labels below expected range (0-59)")
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED - Feeder is working correctly!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = verify_feeder()
    sys.exit(0 if success else 1)
