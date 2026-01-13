# NTU60 Loader Test Files

This directory contains test files to verify the NTU60 YOLO pose loader functionality.

## Test Files

### 1. `test_ntu60_loader_quick.py`
**Quick minimal test** - Run this first to verify basic functionality.

Tests:
- Checks if `data.json` exists
- Initializes the feeder
- Retrieves one sample
- Validates data shape

Usage:
```bash
python test_files/ntu60/test_ntu60_loader_quick.py
```

### 2. `test_feeder_verification.py`
**Comprehensive feeder test** - Verifies train and val feeders work correctly.

Tests:
- Train feeder initialization and sample retrieval
- Val feeder initialization and sample retrieval
- Multiple sample retrieval
- Label distribution validation
- Data shape and format validation

Usage:
```bash
python test_files/ntu60/test_feeder_verification.py
```

### 3. `test_json_loading.py`
**JSON file structure test** - Validates preprocessed JSON files.

Tests:
- JSON file existence and structure
- Required fields (`skeletons`, `metadata`)
- Skeleton format validation (17 joints, [x, y, conf])
- Feeder loading from JSON files

Usage:
```bash
python test_files/ntu60/test_json_loading.py
```

### 4. `test_data_json_loading.py`
**data.json structure test** - Validates the data.json file format.

Tests:
- data.json file structure (`splits` key)
- Split structure validation (train/val)
- Item structure validation (required fields)
- json_path existence check
- Feeder loading from data.json

Usage:
```bash
python test_files/ntu60/test_data_json_loading.py
```

## Prerequisites

Before running the tests, ensure:

1. **Data split file exists:**
   ```bash
   python create_ntu60_split.py --video-dir data/nturgb+d_rgb --output-dir data/ntu60_splits
   ```

2. **Preprocessed JSON files exist (for some tests):**
   ```bash
   python main_multipart_yolo.py --mode preprocess --config config/ntu60_yolo.yaml
   ```

## Running All Tests

Run tests in order (quick test first, then comprehensive):

```bash
# Quick test
python test_files/ntu60/test_ntu60_loader_quick.py

# Comprehensive tests
python test_files/ntu60/test_feeder_verification.py
python test_files/ntu60/test_json_loading.py
python test_files/ntu60/test_data_json_loading.py
```

## Expected Output

All tests should show:
- ✅ Green checkmarks for successful operations
- ❌ Red X for errors
- ⚠️ Yellow warnings for non-critical issues

Successful test output example:
```
✅ Found data.json at data/ntu60_splits/data.json
✅ Feeder initialized successfully
   - Number of samples: 100
✅ Sample retrieved successfully
   - Data shape: (2, 64, 17, 1)
   - Label: 5
🎉 Quick test passed!
```

## Troubleshooting

### Error: data.json not found
- Run `create_ntu60_split.py` to generate the data.json file

### Error: JSON files not found
- Run preprocessing: `python main_multipart_yolo.py --mode preprocess --config config/ntu60_yolo.yaml`

### Error: Feeder has no samples
- Check that `data.json` contains valid items with `json_path` pointing to existing JSON files
- Verify JSON files exist in `data/ntu60_json/train/` and `data/ntu60_json/val/`

### Error: Shape mismatch
- Check that `window_size` matches the expected value (default: 64)
- Verify JSON files contain valid skeleton data
