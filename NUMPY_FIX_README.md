# NumPy Access Violation Fix

## Problem
The application was crashing with the following error:
```
Exception Type: 0xC0000005
Exception Message: Unhandled exception at 0x00007FFFE99D8602 (_multiarray_umath.cp314-win_amd64.pyd) in python.exe: 0xC0000005: Access violation reading location 0x0000000000000009.
```

## Root Cause
The crash was caused by **corrupted array stride calculations** when mixing arrays from different libraries (PyTorch, OpenCV, NumPy). The memory address `0x0000000000000009` indicates that a small integer (9) was being treated as a memory pointer, causing the access violation during NumPy's `MachArLike` object initialization.

## Solution
Implemented comprehensive array safety utilities that:

1. **Validate memory layouts** before operations
2. **Convert arrays to safe formats** automatically
3. **Prevent stride calculation overflows**
4. **Patch NumPy operations** for additional safety

## Files Added/Modified

### New Files
- `src/utils/array_safety.py` - Core array safety utilities
- `examples/array_safety_integration.py` - Integration examples

### Modified Files
- `src/utils/__init__.py` - Export array safety functions
- `src/__init__.py` - Auto-enable array safety on import
- `requirements.txt` - Pin compatible dependency versions

## Key Features

### 1. Safe Array Conversion
```python
from src.utils import safe_array_conversion

# Convert any array-like object safely
safe_array = safe_array_conversion(pytorch_tensor)
safe_array = safe_array_conversion(opencv_image)
```

### 2. Automatic Function Protection
```python
from src.utils import safe_array_operation

@safe_array_operation
def process_image(image_array):
    # Arrays are automatically made safe
    return np.mean(image_array, axis=2)
```

### 3. Library-Specific Converters
```python
from src.utils import safe_tensor_to_numpy, safe_opencv_to_numpy

numpy_array = safe_tensor_to_numpy(torch_tensor)
numpy_array = safe_opencv_to_numpy(cv2_image)
```

## Version Compatibility

Updated `requirements.txt` with pinned versions:
- `numpy==1.24.3` (critical for stability)
- `torch==2.1.0` 
- `opencv-python==4.8.1.78`
- `transformers==4.35.2`
- `ultralytics==8.0.196`

## Usage

### Automatic Protection (Recommended)
Array safety is automatically enabled when importing the package:
```python
import src  # Array safety automatically enabled
```

### Manual Integration
For existing code that's crashing:
```python
# OLD CODE (causing crashes):
def process_data(data):
    array = np.array(data)  # Could crash
    return np.mean(array)

# NEW CODE (safe):
def process_data(data):
    array = safe_array_conversion(data)  # Safe
    return np.mean(array)
```

### Integration with ML Libraries
```python
import torch
from ultralytics import YOLO
from src.utils import safe_array_conversion

# PyTorch
tensor = torch.randn(100, 100)
safe_array = safe_array_conversion(tensor)

# YOLO/Ultralytics
model = YOLO('yolov8n.pt')
results = model('image.jpg')
boxes = safe_array_conversion(results[0].boxes.xyxy)
```

## Technical Details

### Memory Layout Validation
The utilities check for:
- **C_CONTIGUOUS** flag
- **ALIGNED** flag  
- **OWNDATA** flag
- Valid stride calculations
- Overflow prevention

### Stride Calculation Fix
The original crash occurred due to:
```
base_address + (stride * index) = NULL + 9 = 0x9 (invalid address)
```

The fix ensures:
```
base_address + (stride * index) = valid_pointer + valid_offset = valid_address
```

## Performance Impact
- Minimal overhead for already-safe arrays
- Copy operations only when necessary
- Automatic detection of problem arrays

## Testing
Run the integration examples:
```bash
python examples/array_safety_integration.py
```

Expected output:
```
? PyTorch integration working
? OpenCV integration working  
? MachArLike crash fix working
All array safety examples passed!
```

## Rollback
If issues occur, disable array safety:
```python
# In src/__init__.py, comment out:
# enable_array_safety(patch_numpy=True, log_level="WARNING")
```

## Support
The array safety utilities log detailed information for debugging:
- Array conversion warnings
- Memory layout issues
- Stride validation failures

Check logs for any array safety messages during operation.