"""
Check CUDA availability in OpenCV
"""
import cv2

print("=" * 60)
print("CUDA DETECTION CHECK")
print("=" * 60)

# Check if CUDA module exists
if hasattr(cv2, 'cuda'):
    print("\n? OpenCV has CUDA module")
    
    # Check for CUDA device count method
    if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
        print("? getCudaEnabledDeviceCount method exists")
        
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"\nCUDA Device Count: {device_count}")
            
            if device_count > 0:
                print(f"\n? {device_count} CUDA device(s) found!")
                for i in range(device_count):
                    cv2.cuda.setDevice(i)
                    print(f"   Device {i}: Available")
            else:
                print("\n? No CUDA devices found")
                print("\nPossible reasons:")
                print("  1. No NVIDIA GPU in system")
                print("  2. NVIDIA drivers not installed")
                print("  3. CUDA toolkit not installed")
        except Exception as e:
            print(f"\n? Error checking CUDA devices: {e}")
    else:
        print("\n? getCudaEnabledDeviceCount method NOT found")
        print("   OpenCV compiled without CUDA support")
else:
    print("\n? OpenCV does NOT have CUDA module")
    print("   This OpenCV build was not compiled with CUDA support")

print("\n" + "=" * 60)
print("OpenCV Build Information:")
print("=" * 60)
print(cv2.getBuildInformation())

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("\nTo enable CUDA in OpenCV:")
print("1. Uninstall current OpenCV: pip uninstall opencv-python")
print("2. Install CUDA-enabled version:")
print("   pip install opencv-contrib-python")
print("\nOR build OpenCV from source with CUDA enabled")
print("(Complex process - requires CUDA toolkit, CMake, Visual Studio)")
