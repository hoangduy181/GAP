#!/usr/bin/env python
"""Test script to verify NTU2D data setup works correctly"""
import sys
import numpy as np
import torch

# Test 1: Import graph
print("=" * 60)
print("TEST 1: Import Graph")
print("=" * 60)
try:
    from graph.joint17 import Graph
    graph = Graph(labeling_mode='spatial')
    print(f"✅ Graph loaded successfully")
    print(f"   Number of nodes: {graph.num_node}")
    print(f"   Adjacency matrix shape: {graph.A.shape}")
    print(f"   Expected: (3, 17, 17)")
    assert graph.A.shape == (3, 17, 17), f"Expected (3, 17, 17), got {graph.A.shape}"
    print(f"   ✅ Adjacency matrix shape correct!")
except Exception as e:
    print(f"❌ Graph import failed: {e}")
    sys.exit(1)

# Test 2: Import feeder
print("\n" + "=" * 60)
print("TEST 2: Import Feeder")
print("=" * 60)
try:
    from feeders.feeder_ntu_2d import Feeder
    print(f"✅ Feeder imported successfully")
except Exception as e:
    print(f"❌ Feeder import failed: {e}")
    sys.exit(1)

# Test 3: Load data
print("\n" + "=" * 60)
print("TEST 3: Load Data")
print("=" * 60)
try:
    feeder = Feeder(
        data_path='data/ntu2d/ntu60_2d.pkl',
        split='train',
        split_type='xsub',
        window_size=64,
        debug=True  # Use only 100 samples for testing
    )
    print(f"✅ Data loaded successfully")
    print(f"   Number of samples: {len(feeder)}")
    print(f"   Expected: 100 (debug mode)")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Get a sample
print("\n" + "=" * 60)
print("TEST 4: Get Sample")
print("=" * 60)
try:
    data, label, index = feeder[0]
    print('type data', type(feeder[0]))
    print(f"✅ Sample retrieved successfully")
    print(f"   Data shape: {data.shape}")
    print(f"   Expected: (C=2, T=64, V=17, M=1)")
    print(f"   Label: {label}")
    print(f"   Label type: {type(label)}")
    
    assert len(data.shape) == 4, f"Expected 4D tensor, got {len(data.shape)}D"
    assert data.shape[0] == 2, f"Expected C=2, got {data.shape[0]}"
    assert data.shape[2] == 17, f"Expected V=17, got {data.shape[2]}"
    assert data.shape[3] == 1, f"Expected M=1, got {data.shape[3]}"
    print(f"   ✅ Data shape correct!")
except Exception as e:
    print(f"❌ Sample retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test model initialization
print("\n" + "=" * 60)
print("TEST 5: Model Initialization")
print("=" * 60)
try:
    from model.ctrgcn import Model
    model = Model(
        num_class=60,
        num_point=17,
        num_person=1,
        in_channels=2,
        graph='graph.joint17.Graph',
        graph_args={'labeling_mode': 'spatial'}
    )
    print(f"✅ Model initialized successfully")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_data = torch.randn(2, 2, 64, 17, 1)  # (N=2, C=2, T=64, V=17, M=1)
    output = model(batch_data)
    print(f"   Forward pass successful")
    print(f"   Input shape: {batch_data.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output: (2, 60)")
    assert output.shape == (2, 60), f"Expected (2, 60), got {output.shape}"
    print(f"   ✅ Forward pass correct!")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test DataLoader
print("\n" + "=" * 60)
print("TEST 6: DataLoader")
print("=" * 60)
try:
    from torch.utils.data import DataLoader
    loader = DataLoader(feeder, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    data_batch, label_batch, index_batch = batch
    print(f"✅ DataLoader works")
    print(f"   Batch data shape: {data_batch.shape}")
    print(f"   Expected: (N=4, C=2, T=64, V=17, M=1)")
    print(f"   Batch label shape: {label_batch.shape}")
    print(f"   Expected: (4,)")
    assert data_batch.shape[0] == 4, f"Expected batch size 4, got {data_batch.shape[0]}"
    print(f"   ✅ DataLoader correct!")
except Exception as e:
    print(f"❌ DataLoader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe NTU2D 17-joint setup is ready to use!")
print("\nTo run training:")
print("  python main.py --config config/ntu2d/default.yaml")
