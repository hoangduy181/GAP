# Deprecation Issues Found in CTR-GCN Codebase

This document lists all deprecated packages and functions found in the codebase that need to be fixed for compatibility with modern Python and PyTorch versions.

## Critical Issues (Must Fix)

### 1. PyTorch Variable (Deprecated since PyTorch 1.0)

**Status**: ⚠️ **CRITICAL** - Will cause errors in PyTorch 1.0+

**Files Affected**:
- `model/ctrgcn.py` (line 7, 222)
- `model/baseline.py` (line 7, 65)
- `torchlight/torchlight/util.py` (line 15)

**Issue**:
```python
from torch.autograd import Variable  # DEPRECATED
self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
```

**Fix**: Remove Variable wrapper - tensors are Variables by default in modern PyTorch
```python
# OLD (deprecated):
self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

# NEW (correct):
self.A = torch.from_numpy(A.astype(np.float32))
# Note: requires_grad=False is default for torch.from_numpy
```

### 2. Tensor.get_device() (Deprecated)

**Status**: ⚠️ **CRITICAL** - Deprecated in PyTorch 1.0+

**Files Affected**:
- `model/ctrgcn.py` (line 240)
- `model/baseline.py` (line 105)

**Issue**:
```python
A = self.A.cuda(x.get_device())  # DEPRECATED
```

**Fix**: Use `.device` attribute instead
```python
# OLD (deprecated):
A = self.A.cuda(x.get_device())

# NEW (correct):
A = self.A.to(x.device)
```

### 3. YAML Loading (Security Issue)

**Status**: ✅ **FIXED** - Already updated in main.py line 565

**Previous Issue**:
```python
default_arg = yaml.load(f)  # UNSAFE - can execute arbitrary code
```

**Fixed**:
```python
default_arg = yaml.safe_load(f)  # SAFE
```

## Medium Priority Issues

### 4. NumPy Integer Types

**Status**: ⚠️ **WARNING** - May cause issues in NumPy 1.20+

**Files Affected**:
- `feeders/feeder_ucla.py` (line 126) - Actually OK, uses `int` which is correct
- `feeders/tools.py` (line 48-49) - Uses division `/` which is fine for numpy arrays

**Note**: The code uses `astype(int)` which is correct. `np.int` was deprecated, but `int` is fine.

### 5. Division Operator in Python 2 vs 3

**Status**: ✅ **OK** - Code uses numpy arrays, so division works correctly

**Files Affected**:
- `feeders/tools.py` (line 48-49)

**Note**: Using `/` with numpy arrays is fine. The code is Python 3 compatible.

## Low Priority / Style Issues

### 6. Type Checking Style

**Status**: ℹ️ **INFO** - Not deprecated, but could be improved

**Files Affected**:
- `model/ctrgcn.py` (line 88)

**Current**:
```python
if type(kernel_size) == list:  # Works but not Pythonic
```

**Better**:
```python
if isinstance(kernel_size, list):  # More Pythonic
```

### 7. Unused Import

**Status**: ℹ️ **INFO** - Not critical

**Files Affected**:
- `model/ctrgcn.py` (line 2) - `import pdb` (debugging tool, rarely used)

## Summary

### Must Fix Before Running:
1. ✅ YAML loading (FIXED)
2. ❌ PyTorch Variable (3 files)
3. ❌ Tensor.get_device() (2 files)

### Recommended Fixes:
- Use `.to(device)` instead of `.cuda(device)` for better device handling
- Consider using `isinstance()` instead of `type() ==`

## Fix Priority

1. **High**: Fix Variable and get_device() - these will cause runtime errors
2. **Medium**: Review NumPy usage (currently OK)
3. **Low**: Code style improvements

## Testing After Fixes

After fixing the deprecated functions, test:
1. Model initialization
2. Forward pass
3. Training loop
4. Inference
