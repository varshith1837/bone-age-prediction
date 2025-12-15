"""
System Compatibility Check for Bone Age Prediction Project
Run this before starting the project to verify your setup
"""

import sys
import platform
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_python():
    """Check Python version"""
    print_header("Python Version Check")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False

def check_system():
    """Check system information"""
    print_header("System Information")
    print(f"System: {platform.system()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    if platform.system() == "Darwin" and "arm" in platform.machine().lower():
        print("✓ Running on Apple Silicon (M-series chip)")
        return True
    else:
        print("⚠️  Not running on Apple Silicon - MPS may not be available")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Dependency Check")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'timm': 'Timm (PyTorch Image Models)',
        'xgboost': 'XGBoost',
        'sklearn': 'Scikit-learn',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'cv2': 'OpenCV',
        'tqdm': 'TQDM'
    }
    
    missing = []
    installed = []
    
    for module, name in dependencies.items():
        try:
            if module == 'sklearn':
                __import__('sklearn')
            elif module == 'PIL':
                __import__('PIL')
            elif module == 'cv2':
                __import__('cv2')
            else:
                __import__(module)
            installed.append(name)
            print(f"✓ {name}")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} - NOT INSTALLED")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} package(s)")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✓ All {len(installed)} required packages installed")
        return True

def check_pytorch_mps():
    """Check PyTorch MPS availability"""
    print_header("PyTorch MPS Check")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("✓ MPS (Metal Performance Shaders) is available")
            print("  Your M4 Mac can use GPU acceleration!")
            
            # Test MPS
            try:
                x = torch.randn(10, 10, device='mps')
                y = x @ x.T
                print("✓ MPS test successful")
                return True
            except Exception as e:
                print(f"✗ MPS test failed: {e}")
                return False
        else:
            print("✗ MPS not available")
            print("  Training will use CPU (slower)")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_dataset():
    """Check if dataset is properly set up"""
    print_header("Dataset Check")
    
    dataset_root = Path("dataset")
    
    if not dataset_root.exists():
        print("✗ dataset/ directory not found")
        print("  Create it and add your data")
        return False
    
    train_dir = dataset_root / "boneage-training-dataset"
    test_dir = dataset_root / "boneage-test-dataset"
    train_csv = dataset_root / "boneage-training-dataset.csv"
    test_csv = dataset_root / "boneage-test-dataset.csv"
    
    checks = [
        (train_dir, "Training images directory"),
        (test_dir, "Test images directory"),
        (train_csv, "Training CSV"),
        (test_csv, "Test CSV")
    ]
    
    all_ok = True
    for path, description in checks:
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob("*.png"))) + len(list(path.glob("*.jpg")))
                print(f"✓ {description}: {count} images")
            else:
                print(f"✓ {description}")
        else:
            print(f"✗ {description} not found")
            all_ok = False
    
    return all_ok

def check_disk_space():
    """Check available disk space"""
    print_header("Disk Space Check")
    
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    print(f"Free disk space: {free_gb:.2f} GB")
    
    if free_gb > 10:
        print("✓ Sufficient disk space (10+ GB recommended)")
        return True
    elif free_gb > 5:
        print("⚠️  Low disk space (5-10 GB)")
        print("  Should be okay, but consider freeing up space")
        return True
    else:
        print("✗ Insufficient disk space (< 5 GB)")
        print("  You need at least 5 GB free")
        return False

def check_memory():
    """Check RAM"""
    print_header("Memory Check")
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)
        
        print(f"Total RAM: {total_gb:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB")
        
        if total_gb >= 16:
            print("✓ RAM is sufficient (16+ GB)")
            return True
        elif total_gb >= 8:
            print("⚠️  RAM is limited (8-16 GB)")
            print("  Consider reducing batch size in config.py")
            return True
        else:
            print("✗ Insufficient RAM (< 8 GB)")
            print("  8GB minimum required")
            return False
            
    except ImportError:
        print("⚠️  psutil not installed - cannot check RAM")
        print("  Install: pip install psutil")
        return True

def main():
    """Run all checks"""
    print("\n" + "="*70)
    print(" "*15 + "SYSTEM COMPATIBILITY CHECK")
    print(" "*10 + "Bone Age Prediction Project")
    print("="*70)
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python()))
    results.append(("System Info", check_system()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("PyTorch MPS", check_pytorch_mps()))
    results.append(("Dataset", check_dataset()))
    results.append(("Disk Space", check_disk_space()))
    results.append(("Memory", check_memory()))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n" + "="*70)
        print("  ✓ ALL CHECKS PASSED - SYSTEM IS READY!")
        print("  Run: python main.py")
        print("="*70 + "\n")
    elif passed >= total - 1:
        print("\n" + "="*70)
        print("  ⚠️  SYSTEM IS MOSTLY READY")
        print("  Address the failed check(s) if possible")
        print("  You may proceed with: python main.py")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("  ✗ SYSTEM NOT READY")
        print("  Please address the failed checks before proceeding")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()