"""
Quick test to verify project setup
Run: python test_setup.py
"""
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Test imports
try:
    from models import create_model, RAVENFeatureExtractor, TransformerReasoner
    print("✓ Models imported successfully")
except ImportError as e:
    print(f"✗ Model import error: {e}")

try:
    from utils import RAVENDataset, ModelEvaluator
    print("✓ Utils imported successfully")
except ImportError as e:
    print(f"✗ Utils import error: {e}")

try:
    from config import DEVICE, BATCH_SIZE, EPOCHS
    print(f"✓ Config loaded (device={DEVICE}, batch_size={BATCH_SIZE})")
except ImportError as e:
    print(f"✗ Config import error: {e}")

# Test model creation
try:
    model = create_model('transformer', pretrained_encoder=False)
    dummy_input = torch.randn(2, 16, 160, 160)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Model forward pass works (output shape: {output.shape})")
except Exception as e:
    print(f"✗ Model test failed: {e}")

# Check for data (supports RAVEN and I-RAVEN datasets)
from pathlib import Path
data_dirs = [
    # RAVEN datasets
    Path("./data/raven_medium"),
    Path("./data/raven_small"),
    Path("./data/raven_large"),
    # I-RAVEN datasets (bias-corrected)
    Path("./data/iraven_medium"),
    Path("./data/iraven_small"),
    Path("./data/iraven_large"),
    # Colab paths
    Path("/content/drive/MyDrive/raven_medium"),
    Path("/content/drive/MyDrive/raven_small"),
    Path("/content/drive/MyDrive/iraven_medium"),
]
data_found = False
for d in data_dirs:
    if d.exists():
        npz_count = len(list(d.rglob("*.npz")))
        if npz_count > 0:
            dataset_type = "I-RAVEN" if "iraven" in str(d) else "RAVEN"
            print(f"✓ {dataset_type} data found: {d} ({npz_count} files)")
            data_found = True
if not data_found:
    print("✗ No data found.")
    print("  For RAVEN (original):       ./setup.sh")
    print("  For I-RAVEN (bias-free):    ./setup_iraven.sh")

print("\n" + "="*50)
print("Setup test complete!")
print("="*50)

