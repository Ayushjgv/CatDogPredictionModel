"""
Diagnostic script to verify GPU availability, CUDA version, device specs,
and run a tensor operation test on the NVIDIA GPU.
"""

import sys
import time

def check_gpu():
    print("=" * 60)
    print("           GPU & PyTorch Diagnostic Checker")
    print("=" * 60)
    
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    
    try:
        import torch
    except ImportError:
        print("\n[ERROR] PyTorch is not installed in this environment.")
        print("Please run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return False
        
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA compiled with: {torch.version.cuda}")
    print(f"cuDNN version     : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available    : {cuda_available}")
    
    if not cuda_available:
        print("\n[WARNING] CUDA is NOT available to PyTorch!")
        print("Possible causes:")
        print(" 1. The CPU-only version of PyTorch was installed.")
        print(" 2. NVIDIA drivers are missing or require a reboot.")
        return False
        
    device_count = torch.cuda.device_count()
    print(f"GPU Count         : {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"\n--- GPU Device [{i}]: {props.name} ---")
        print(f"  Compute Capability : {props.major}.{props.minor}")
        print(f"  Total VRAM         : {total_mem_gb:.2f} GB")
        print(f"  Multi-processors   : {props.multi_processor_count}")
        
    # Run a tensor computation test on GPU
    print("\n" + "-" * 60)
    print("Running tensor matrix multiplication test on GPU (cuda:0)...")
    try:
        device = torch.device("cuda:0")
        start_time = time.time()
        
        # Create two 2048 x 2048 float32 matrices on GPU
        x = torch.randn(2048, 2048, device=device, dtype=torch.float32)
        y = torch.randn(2048, 2048, device=device, dtype=torch.float32)
        
        # Warmup and synchronize
        z = torch.matmul(x, y)
        torch.cuda.synchronize(device)
        elapsed = time.time() - start_time
        
        mem_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        mem_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        
        print(f"Tensor computation successful in {elapsed:.4f}s!")
        print(f"GPU Memory Allocated: {mem_allocated_mb:.2f} MB")
        print(f"GPU Memory Reserved : {mem_reserved_mb:.2f} MB")
        print("=" * 60)
        print(" [SUCCESS] GPU is fully configured and ready for model training!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"[ERROR] GPU computation failed: {e}")
        return False

if __name__ == "__main__":
    check_gpu()
