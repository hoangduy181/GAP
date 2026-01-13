#!/usr/bin/env python3
"""
Test script for NTU60 data.json loading

Tests loading data from the data.json file with splits structure.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

def test_data_json_loading():
    """Test loading from data.json file"""
    print("=" * 70)
    print("NTU60 data.json Loading Test")
    print("=" * 70)
    
    # Check if data.json exists
    data_json_path = "data/ntu60_splits/data.json"
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        print("\nPlease create data.json first:")
        print("  python create_ntu60_split.py --video-dir data/nturgb+d_rgb --output-dir data/ntu60_splits")
        return False
    
    print(f"✅ Found data.json at {data_json_path}")
    
    # Load and validate data.json structure
    try:
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
        
        print(f"\n📊 Validating data.json structure...")
        
        # Check for 'splits' key
        if 'splits' not in data_json:
            print(f"   ❌ ERROR: Missing 'splits' key in data.json")
            return False
        
        splits = data_json['splits']
        if not isinstance(splits, list):
            print(f"   ❌ ERROR: 'splits' should be a list")
            return False
        
        print(f"   ✅ Found {len(splits)} split(s)")
        
        # Check each split
        for split_info in splits:
            split_name = split_info.get('split', 'unknown')
            items = split_info.get('items', [])
            
            print(f"\n   📂 Split: {split_name}")
            print(f"      - Number of items: {len(items)}")
            
            if len(items) == 0:
                print(f"      ⚠️  WARNING: No items in {split_name} split")
                continue
            
            # Check first item structure
            first_item = items[0]
            required_fields = ['video_name', 'video_path', 'label_action']
            for field in required_fields:
                if field not in first_item:
                    print(f"      ❌ ERROR: Missing required field '{field}' in {split_name} split")
                    return False
            
            # Check if json_path exists (if provided)
            if 'json_path' in first_item:
                json_path = first_item['json_path']
                if os.path.exists(json_path):
                    print(f"      ✅ json_path exists: {os.path.basename(json_path)}")
                else:
                    print(f"      ⚠️  WARNING: json_path not found: {json_path}")
            
            # Count items with json_path
            items_with_json = sum(1 for item in items if 'json_path' in item and os.path.exists(item.get('json_path', '')))
            print(f"      - Items with valid json_path: {items_with_json}/{len(items)}")
            
            # Check label range
            labels = [item.get('label_action', -1) for item in items]
            if labels:
                print(f"      - Label range: {min(labels)} to {max(labels)}")
                if max(labels) >= 60:
                    print(f"      ⚠️  WARNING: Labels exceed expected range (0-59)")
        
        print(f"\n✅ data.json structure validation passed!")
        
        # Test loading with feeder
        print(f"\n📊 Testing feeder loading from data.json...")
        try:
            from feeders.feeder_yolo_pose_ntu60 import Feeder
            
            # Test train split
            print(f"\n   Testing TRAIN split...")
            train_feeder = Feeder(
                data_path="data/ntu60_json",
                split="train",
                data_json_path=data_json_path,
                window_size=64,
                debug=True,
                random_choose=False,
                repeat=1,
                use_cache=False
            )
            print(f"   ✅ Train feeder loaded {len(train_feeder)} samples")
            
            if len(train_feeder) > 0:
                data, label, index = train_feeder[0]
                print(f"      - Sample shape: {data.shape}")
                print(f"      - Sample label: {label}")
            
            # Test val split
            print(f"\n   Testing VAL split...")
            val_feeder = Feeder(
                data_path="data/ntu60_json",
                split="val",
                data_json_path=data_json_path,
                window_size=64,
                debug=True,
                random_choose=False,
                repeat=1,
                use_cache=False
            )
            print(f"   ✅ Val feeder loaded {len(val_feeder)} samples")
            
            if len(val_feeder) > 0:
                data, label, index = val_feeder[0]
                print(f"      - Sample shape: {data.shape}")
                print(f"      - Sample label: {label}")
            
            print(f"\n✅ Feeder can successfully load from data.json!")
            return True
            
        except Exception as e:
            print(f"❌ ERROR: Feeder loading failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in data.json: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Failed to process data.json: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_data_json_loading()
    sys.exit(0 if success else 1)
