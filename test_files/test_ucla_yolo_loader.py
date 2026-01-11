#!/usr/bin/env python3
"""
Test script for UCLA YOLO Pose Loader (feeder_yolo_pose_ucla.py)

This script tests:
1. Data loading from data.json
2. Data loading from JSON files directly
3. Sample retrieval and format validation
4. Label extraction
5. Data augmentation options
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project root to path (go up one level from test_files/)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory for relative paths to work
os.chdir(project_root)

from feeders.feeder_yolo_pose_ucla import Feeder


def test_data_json_loading():
    """Test loading data from data.json file"""
    print("=" * 70)
    print("TEST 1: Loading from data.json")
    print("=" * 70)
    
    data_json_path = "data/yolo_ucla/data.json"
    
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        print("   Please run preprocessing first or check the path.")
        return False
    
    try:
        # Test train split
        print(f"\n📂 Testing train split...")
        train_feeder = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="train",
            split="train",
            data_json_path=data_json_path,
            window_size=52,
            debug=False,
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False  # Don't use cache for testing
        )
        
        print(f"✅ Train feeder initialized successfully")
        print(f"   - Number of samples: {len(train_feeder)}")
        print(f"   - Number of labels: {len(set(train_feeder.label_list))}")
        print(f"   - Label range: {min(train_feeder.label_list)} to {max(train_feeder.label_list)}")
        
        # Test val split
        print(f"\n📂 Testing val split...")
        val_feeder = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="val",
            split="test",
            data_json_path=data_json_path,
            window_size=52,
            debug=False,
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        print(f"✅ Val feeder initialized successfully")
        print(f"   - Number of samples: {len(val_feeder)}")
        print(f"   - Number of labels: {len(set(val_feeder.label_list))}")
        print(f"   - Label range: {min(val_feeder.label_list)} to {max(val_feeder.label_list)}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load from data.json")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_retrieval():
    """Test retrieving and validating samples"""
    print("\n" + "=" * 70)
    print("TEST 2: Sample Retrieval and Format Validation")
    print("=" * 70)
    
    data_json_path = "data/yolo_ucla/data.json"
    
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        return False
    
    try:
        feeder = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="train",
            split="train",
            data_json_path=data_json_path,
            window_size=52,
            debug=True,  # Use debug mode for faster testing
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        if len(feeder) == 0:
            print("❌ ERROR: Feeder has no samples")
            return False
        
        print(f"\n📊 Testing sample retrieval (using first 5 samples)...")
        
        for idx in range(min(5, len(feeder))):
            try:
                data, label, index = feeder[idx]
                
                # Validate data format
                assert isinstance(data, np.ndarray), f"Data should be numpy array, got {type(data)}"
                assert data.shape == (2, 52, 17, 1), f"Expected shape (2, 52, 17, 1), got {data.shape}"
                assert data.dtype == np.float32, f"Data should be float32, got {data.dtype}"
                
                # Validate label
                assert isinstance(label, (int, np.integer)), f"Label should be int, got {type(label)}"
                assert 0 <= label < 10, f"Label should be 0-9, got {label}"
                
                # Validate index
                assert isinstance(index, (int, np.integer)), f"Index should be int, got {type(index)}"
                
                # Get sample name from feeder if available
                sample_name = "N/A"
                if hasattr(feeder, 'sample_name') and len(feeder.sample_name) > index:
                    sample_name = feeder.sample_name[index]
                
                print(f"   ✅ Sample {idx}: shape={data.shape}, label={label}, index={index}, name={sample_name[:30] if isinstance(sample_name, str) else 'N/A'}...")
                
            except Exception as e:
                print(f"   ❌ ERROR retrieving sample {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n✅ All sample retrievals successful!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve samples")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_augmentation():
    """Test data augmentation options"""
    print("\n" + "=" * 70)
    print("TEST 3: Data Augmentation Options")
    print("=" * 70)
    
    data_json_path = "data/yolo_ucla/data.json"
    
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        return False
    
    try:
        # Test with random_choose enabled
        print(f"\n📊 Testing with random_choose=True...")
        feeder_random = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="train",
            split="train",
            data_json_path=data_json_path,
            window_size=52,
            debug=True,
            random_choose=True,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        sample1 = feeder_random[0]
        sample2 = feeder_random[0]  # Same index, should be different with random_choose
        
        if not np.array_equal(sample1[0], sample2[0]):
            print(f"   ✅ random_choose is working (samples differ)")
        else:
            print(f"   ⚠️  WARNING: random_choose samples are identical (might be expected)")
        
        # Test with repeat > 1
        print(f"\n📊 Testing with repeat=3...")
        feeder_repeat = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="train",
            split="train",
            data_json_path=data_json_path,
            window_size=52,
            debug=True,
            random_choose=False,
            repeat=3,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        print(f"   ✅ Feeder with repeat=3 has {len(feeder_repeat)} samples")
        print(f"   (Original: {len(feeder_random)}, Expected: {len(feeder_random) * 3})")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to test augmentation")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_label_distribution():
    """Test label distribution across splits"""
    print("\n" + "=" * 70)
    print("TEST 4: Label Distribution")
    print("=" * 70)
    
    data_json_path = "data/yolo_ucla/data.json"
    
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        return False
    
    try:
        # Load data.json to check distribution
        with open(data_json_path, 'r') as f:
            data_json = json.load(f)
        
        train_data = data_json.get('train', [])
        val_data = data_json.get('val', [])
        
        # Count labels in train
        train_labels = [item['label'] for item in train_data]
        train_label_counts = {}
        for label in train_labels:
            train_label_counts[label] = train_label_counts.get(label, 0) + 1
        
        # Count labels in val
        val_labels = [item['label'] for item in val_data]
        val_label_counts = {}
        for label in val_labels:
            val_label_counts[label] = val_label_counts.get(label, 0) + 1
        
        print(f"\n📊 Train split label distribution:")
        for label in sorted(train_label_counts.keys()):
            count = train_label_counts[label]
            print(f"   Label {label}: {count} samples")
        
        print(f"\n📊 Val split label distribution:")
        for label in sorted(val_label_counts.keys()):
            count = val_label_counts[label]
            print(f"   Label {label}: {count} samples")
        
        # Check if all labels are present
        all_labels = set(train_label_counts.keys()) | set(val_label_counts.keys())
        expected_labels = set(range(10))  # UCLA has 10 classes
        
        if all_labels == expected_labels:
            print(f"\n✅ All 10 action classes are present")
        else:
            missing = expected_labels - all_labels
            print(f"\n⚠️  WARNING: Missing labels: {missing}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to check label distribution")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_json_file_loading():
    """Test loading from JSON files directly (fallback method)"""
    print("\n" + "=" * 70)
    print("TEST 5: Direct JSON File Loading (Fallback)")
    print("=" * 70)
    
    train_dir = "data/yolo_ucla_json/train"
    
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Train directory not found at {train_dir}")
        return False
    
    # Count JSON files
    json_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
    
    if len(json_files) == 0:
        print(f"⚠️  WARNING: No JSON files found in {train_dir}")
        return False
    
    # Filter out metadata/index files (these don't have skeleton data)
    metadata_files = ['train_data_dict.json', 'val_data_dict.json', 'data.json', '_data_dict.json']
    skeleton_files = [f for f in json_files if not any(f.endswith(mf) or mf in f for mf in metadata_files)]
    
    if len(skeleton_files) == 0:
        print(f"⚠️  WARNING: No skeleton data files found (only metadata files)")
        return False
    
    print(f"\n📂 Found {len(json_files)} JSON files in {train_dir}")
    print(f"   - Metadata files: {len(json_files) - len(skeleton_files)}")
    print(f"   - Skeleton data files: {len(skeleton_files)}")
    
    # Test loading a few skeleton JSON files directly
    print(f"\n📊 Testing direct JSON file loading...")
    for i, json_file in enumerate(skeleton_files[:3]):
        json_path = os.path.join(train_dir, json_file)
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Validate structure - actual format uses 'skeletons' (plural)
            assert 'skeletons' in data, f"Missing 'skeletons' field in {json_file}"
            
            # Check that skeletons is a list
            assert isinstance(data['skeletons'], list), f"'skeletons' should be a list in {json_file}"
            
            print(f"   ✅ {json_file}: Valid JSON structure ({len(data['skeletons'])} frames)")
            
        except Exception as e:
            print(f"   ❌ ERROR loading {json_file}: {str(e)}")
            return False
    
    return True


def test_dataloader_integration():
    """Test integration with PyTorch DataLoader"""
    print("\n" + "=" * 70)
    print("TEST 6: PyTorch DataLoader Integration")
    print("=" * 70)
    
    data_json_path = "data/yolo_ucla/data.json"
    
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        return False
    
    try:
        from torch.utils.data import DataLoader
        
        feeder = Feeder(
            data_path="data/yolo_ucla_json",
            label_path="train",
            split="train",
            data_json_path=data_json_path,
            window_size=52,
            debug=True,  # Use debug mode for faster testing
            random_choose=False,
            repeat=1,
            yolo_model_path="yolo11n-pose.pt",
            use_cache=False
        )
        
        dataloader = DataLoader(
            feeder,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False
        )
        
        print(f"\n📊 Testing DataLoader with batch_size=2...")
        
        # Get one batch
        for batch_idx, (data, label, index) in enumerate(dataloader):
            # Validate batch format
            assert isinstance(data, torch.Tensor), f"Data should be torch.Tensor, got {type(data)}"
            assert data.shape[0] == 2, f"Batch size should be 2, got {data.shape[0]}"
            assert data.shape == (2, 2, 52, 17, 1), f"Expected shape (2, 2, 52, 17, 1), got {data.shape}"
            
            assert isinstance(label, torch.Tensor), f"Label should be torch.Tensor, got {type(label)}"
            assert label.shape == (2,), f"Label shape should be (2,), got {label.shape}"
            
            assert isinstance(index, torch.Tensor), f"Index should be torch.Tensor, got {type(index)}"
            assert index.shape == (2,), f"Index shape should be (2,), got {index.shape}"
            
            print(f"   ✅ Batch {batch_idx}: data shape={data.shape}, label shape={label.shape}, index shape={index.shape}")
            
            # Only test first batch
            break
        
        print(f"\n✅ DataLoader integration successful!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed DataLoader integration")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("UCLA YOLO Pose Loader Test Suite")
    print("=" * 70)
    print()
    
    # Check if required files exist
    data_json_path = "data/yolo_ucla/data.json"
    train_dir = "data/yolo_ucla_json/train"
    
    if not os.path.exists(data_json_path):
        print(f"❌ ERROR: data.json not found at {data_json_path}")
        print("   Please run preprocessing first:")
        print("   python main_multipart_yolo_ucla.py --mode process --config config/yolo_ucla/data_config.yaml")
        return 1
    
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Train directory not found at {train_dir}")
        print("   Please run preprocessing first:")
        print("   python main_multipart_yolo_ucla.py --mode process --config config/yolo_ucla/data_config.yaml")
        return 1
    
    # Run tests
    tests = [
        ("Data JSON Loading", test_data_json_loading),
        ("Sample Retrieval", test_sample_retrieval),
        ("Data Augmentation", test_data_augmentation),
        ("Label Distribution", test_label_distribution),
        ("JSON File Loading", test_json_file_loading),
        ("DataLoader Integration", test_dataloader_integration),
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
