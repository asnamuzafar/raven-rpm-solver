#!/bin/bash
# ==============================================
# RAVEN RPM Solver - Setup Script
# ==============================================
# This script sets up the entire project from scratch:
# 1. Creates virtual environment
# 2. Installs dependencies
# 3. Clones and patches RAVEN dataset generator
# 4. Generates the dataset
# 5. Verifies the setup
#
# Usage:
#   ./setup.sh              # Default: generates medium dataset (2000 samples)
#   ./setup.sh small        # Small dataset (200 samples) - for quick testing
#   ./setup.sh medium       # Medium dataset (2000 samples) - recommended
#   ./setup.sh large        # Large dataset (10000 samples) - best performance
# ==============================================

set -e  # Exit on error

# Parse dataset size argument
DATASET_SIZE="${1:-medium}"  # Default to medium

case "$DATASET_SIZE" in
    small)
        NUM_SAMPLES=200
        DATASET_NAME="raven_small"
        ;;
    medium)
        NUM_SAMPLES=2000
        DATASET_NAME="raven_medium"
        ;;
    large)
        NUM_SAMPLES=10000
        DATASET_NAME="raven_large"
        ;;
    *)
        echo "Unknown dataset size: $DATASET_SIZE"
        echo "Usage: ./setup.sh [small|medium|large]"
        exit 1
        ;;
esac

echo "=============================================="
echo "RAVEN RPM Solver - Setup Script"
echo "=============================================="
echo "Dataset size: $DATASET_SIZE ($NUM_SAMPLES samples per config)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Create virtual environment
echo -e "\n${YELLOW}[1/5] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Step 2: Install dependencies
echo -e "\n${YELLOW}[2/5] Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Clone and patch RAVEN repository
echo -e "\n${YELLOW}[3/5] Setting up RAVEN dataset generator...${NC}"
if [ ! -d "RAVEN" ]; then
    git clone https://github.com/WellyZhang/RAVEN.git
    echo -e "${GREEN}✓ RAVEN repository cloned${NC}"
else
    echo -e "${GREEN}✓ RAVEN repository already exists${NC}"
fi

cd RAVEN

# Convert Python 2 to Python 3
echo "  Converting to Python 3..."
python -m lib2to3 -w src/ > /dev/null 2>&1 || true

# Apply patches
echo "  Applying patches..."

# Fix scipy import and Python 2/3 compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/AoT.py 2>/dev/null || true
    sed -i '' 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/sampling.py 2>/dev/null || true
    sed -i '' "s/return ET.tostring(data)/return ET.tostring(data, encoding='unicode')/g" src/dataset/serialize.py 2>/dev/null || true
    sed -i '' 's/range(min_level, max_level + 1)/range(int(min_level), int(max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
    sed -i '' 's/range(self.min_level, self.max_level + 1)/range(int(self.min_level), int(self.max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
else
    # Linux (including Colab)
    sed -i 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/AoT.py 2>/dev/null || true
    sed -i 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/sampling.py 2>/dev/null || true
    sed -i "s/return ET.tostring(data)/return ET.tostring(data, encoding='unicode')/g" src/dataset/serialize.py 2>/dev/null || true
    sed -i 's/range(min_level, max_level + 1)/range(int(min_level), int(max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
    sed -i 's/range(self.min_level, self.max_level + 1)/range(int(self.min_level), int(self.max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
fi

# Fix Python 2 print statements that lib2to3 may have missed
echo "  Fixing Python 2 print statements..."
cat > /tmp/fix_print.py << 'PYEOF'
import os
import re

def fix_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        for line in lines:
            # Check if line has Python 2 style print (without parentheses)
            # Pattern: print followed by space and quote, but not print(
            if re.match(r'\s*print\s+["\']', line) and 'print(' not in line:
                # Replace: print "..." -> print("...")
                # Handle the whole line by wrapping print content in parentheses
                new_line = re.sub(r'(\s*)print\s+(.+)$', r'\1print(\2)', line)
                new_lines.append(new_line)
                modified = True
            else:
                new_lines.append(line)
        
        if modified:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            print(f"  Fixed: {filepath}")
    except Exception as e:
        print(f"  Error: {filepath}: {e}")

for root, dirs, files in os.walk('src/'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
PYEOF
python /tmp/fix_print.py 2>/dev/null || true

# Fix absolute imports to relative imports (Python 2 to Python 3 package structure)
echo "  Fixing imports to relative imports..."
cat > /tmp/fix_imports.py << 'PYEOF'
import os
import re

# Modules that exist in src/dataset/ and need relative imports
dataset_modules = [
    'build_tree', 'const', 'rendering', 'Rule', 'sampling', 
    'serialize', 'solver', 'AoT', 'Component', 'constraints',
    'Entity', 'Attribute', 'Position', 'Structure', 'api'
]

def fix_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        
        for module in dataset_modules:
            # from module import ... -> from .module import ...
            pattern = rf'^(from\s+)({module})(\s+import)'
            replacement = rf'\1.\2\3'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            # import module -> from . import module (for dataset submodules)
            pattern2 = rf'^(import\s+)({module})(\s*$)'
            replacement2 = rf'from . \1\2\3'
            content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  Fixed imports: {filepath}")
    except Exception as e:
        print(f"  Error: {filepath}: {e}")

# Fix imports in dataset directory
for file in os.listdir('src/dataset/'):
    if file.endswith('.py'):
        fix_imports(os.path.join('src/dataset/', file))
PYEOF
python /tmp/fix_imports.py 2>/dev/null || true

# Fix Python 3 range() not being a list (needed for .pop(), .index(), etc.)
echo "  Fixing range() to list(range())..."
cat > /tmp/fix_range.py << 'PYEOF'
import re

files_to_fix = [
    'src/dataset/Rule.py',
    'src/dataset/Attribute.py',
    'src/dataset/sampling.py'
]

for filepath in files_to_fix:
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        
        # Pattern: variable = range(...) where variable is later used with list methods
        # Convert: range(x, y) -> list(range(x, y))
        # But avoid double-wrapping: list(range(...))
        
        # Match range() that is NOT already wrapped in list()
        # This handles assignments like: all_value_levels = range(...)
        content = re.sub(
            r'(\s*=\s*)range\(([^)]+)\)(\s*$|\s*#)',
            r'\1list(range(\2))\3',
            content,
            flags=re.MULTILINE
        )
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  Fixed range(): {filepath}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  Error: {filepath}: {e}")
PYEOF
python /tmp/fix_range.py 2>/dev/null || true

# Fix RLE encoding function
cat > /tmp/rle_fix.py << 'EOF'
import re

with open('src/dataset/api.py', 'r') as f:
    content = f.read()

old_func = r'def rle_encode\(img\):.*?return "\[" \+ ",".join\(str\(x\) for x in runs\) \+ "\]"'
new_func = '''def rle_encode(img):
    m = np.asarray(img).astype(np.uint8).reshape(-1)
    z = np.concatenate([[0], m, [0]])
    runs = np.where(z[1:] != z[:-1])[0] + 1
    if runs.size % 2 == 1:
        runs = np.concatenate([runs, [runs[-1]]])
    runs[1::2] -= runs[::2]
    return "[" + ",".join(str(int(x)) for x in runs) + "]"'''

content = re.sub(old_func, new_func, content, flags=re.DOTALL)

with open('src/dataset/api.py', 'w') as f:
    f.write(content)
EOF
python /tmp/rle_fix.py 2>/dev/null || true

echo -e "${GREEN}✓ Patches applied${NC}"

cd "$SCRIPT_DIR"

# Step 4: Generate dataset
echo -e "\n${YELLOW}[4/5] Generating RAVEN dataset '$DATASET_NAME' (this may take a few minutes)...${NC}"
mkdir -p "data/$DATASET_NAME"

cd RAVEN
PYTHONPATH=src python -m dataset.main --num-samples "$NUM_SAMPLES" --save-dir "../data/$DATASET_NAME"
cd "$SCRIPT_DIR"

# Count generated files
NUM_FILES=$(find "data/$DATASET_NAME" -name "*.npz" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Dataset generated: ${NUM_FILES} puzzle files in data/$DATASET_NAME${NC}"

# Step 5: Verify setup
echo -e "\n${YELLOW}[5/5] Verifying setup...${NC}"
python test_setup.py

echo -e "\n${GREEN}=============================================="
echo "SETUP COMPLETE!"
echo "==============================================${NC}"
echo ""
echo "Dataset: data/$DATASET_NAME ($NUM_FILES puzzles)"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Train models:         python train.py --data_dir ./data/$DATASET_NAME --epochs 30"
echo "  3. Evaluate:             python evaluate.py --data_dir ./data/$DATASET_NAME"
echo "  4. Run simulator:        streamlit run raven_simulator.py"
echo ""
echo "Tip: For better model performance, use a larger dataset:"
echo "     ./setup.sh large   # Generates 70,000 puzzles (takes ~30 min)"
echo ""

