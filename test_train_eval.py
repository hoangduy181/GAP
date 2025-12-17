#!/usr/bin/env python3
"""
Simple test script to verify training and evaluation work correctly.
This script runs a minimal training session (3 epochs) and then evaluates.
"""

import os
import sys
import subprocess
import glob
import torch

def check_cuda():
    """Check CUDA availability"""
    print("=" * 50)
    print("Checking CUDA availability...")
    print("=" * 50)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print()

def check_data_path(config_path):
    """Check if data path exists in config"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_path = config.get('train_feeder_args', {}).get('data_path', '')
    test_path = config.get('test_feeder_args', {}).get('data_path', '')
    
    print("=" * 50)
    print("Checking data paths...")
    print("=" * 50)
    print(f"Train data path: {train_path}")
    print(f"  Exists: {os.path.exists(train_path)}")
    print(f"Test data path: {test_path}")
    print(f"  Exists: {os.path.exists(test_path)}")
    print()
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("WARNING: Data files not found!")
        print("Please update the data_path in the config file.")
        print("You can use debug mode which uses minimal data.")
        return False
    return True

def run_training(config_path, work_dir, device='0'):
    """Run training"""
    print("=" * 50)
    print("Step 1: Running training (3 epochs)...")
    print("=" * 50)
    
    cmd = [
        sys.executable, 'main_multipart_ntu.py',
        '--config', config_path,
        '--model', 'model.ctrgcn.Model_lst_4part',
        '--work-dir', work_dir,
        '--phase', 'train',
        '--num-epoch', '3',
        '--save-interval', '1',
        '--save-epoch', '1',
        '--num-worker', '2',
        '--device', device,
        '--print-log', 'True'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("ERROR: Training failed!")
        return None
    return work_dir

def find_model_file(work_dir):
    """Find the saved model file"""
    print("=" * 50)
    print("Step 2: Finding saved model...")
    print("=" * 50)
    
    pattern = os.path.join(work_dir, 'runs-*.pt')
    model_files = glob.glob(pattern)
    
    if not model_files:
        print("ERROR: No model file found!")
        return None
    
    # Get the most recent model file
    model_file = max(model_files, key=os.path.getmtime)
    print(f"Found model: {model_file}")
    print()
    return model_file

def run_evaluation(config_path, work_dir, model_file, device='0'):
    """Run evaluation"""
    print("=" * 50)
    print("Step 3: Running evaluation...")
    print("=" * 50)
    
    cmd = [
        sys.executable, 'main_multipart_ntu.py',
        '--config', config_path,
        '--model', 'model.ctrgcn.Model_lst_4part',
        '--work-dir', work_dir,
        '--phase', 'test',
        '--weights', model_file,
        '--device', device,
        '--print-log', 'True'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("ERROR: Evaluation failed!")
        return False
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test training and evaluation')
    parser.add_argument('--config', type=str, default='config/test_minimal.yaml',
                        help='Path to config file (default: config/test_minimal.yaml)')
    parser.add_argument('--work-dir', type=str, default='work_dir/test_run',
                        help='Work directory for test run (default: work_dir/test_run)')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device(s) to use (default: 0)')
    args = parser.parse_args()
    
    # Set environment variable for CUDA memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    config_path = args.config
    work_dir = args.work_dir
    
    # Clean up previous test run
    if os.path.exists(work_dir):
        print(f"Cleaning up previous test run: {work_dir}")
        import shutil
        shutil.rmtree(work_dir)
        print()
    
    # Check CUDA
    check_cuda()
    
    # Check data paths (informational, don't fail if missing)
    check_data_path(config_path)
    
    # Run training
    result_dir = run_training(config_path, work_dir, args.device)
    if result_dir is None:
        sys.exit(1)
    
    # Find model
    model_file = find_model_file(work_dir)
    if model_file is None:
        sys.exit(1)
    
    # Run evaluation
    success = run_evaluation(config_path, work_dir, model_file, args.device)
    if not success:
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("Test completed successfully!")
    print("=" * 50)
    print(f"Model saved at: {model_file}")
    print(f"Work directory: {work_dir}")

if __name__ == '__main__':
    main()

