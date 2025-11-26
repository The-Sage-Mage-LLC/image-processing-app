"""
Check PyTorch CUDA availability
"""
import torch

print("=" * 60)
print("PYTORCH CUDA DETECTION CHECK")
print("=" * 60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n? PyTorch CUDA is available!")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print(f"  Current Device: {torch.cuda.current_device()}")
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    
    # Test GPU memory
    print(f"\n  GPU Memory:")
    print(f"    Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"    Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # Quick test
    print(f"\n  Running quick GPU test...")
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = x @ y
    print(f"  ? GPU computation test passed!")
    
else:
    print(f"\n? PyTorch CUDA is NOT available")
    print(f"\nPossible reasons:")
    print(f"  1. No NVIDIA GPU in system")
    print(f"  2. NVIDIA drivers not installed")
    print(f"  3. PyTorch CPU-only version installed")

print("\n" + "=" * 60)
