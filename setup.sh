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
# ==============================================

set -e  # Exit on error

echo "=============================================="
echo "RAVEN RPM Solver - Setup Script"
echo "=============================================="

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

# Fix scipy import
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/AoT.py 2>/dev/null || true
    sed -i '' 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/sampling.py 2>/dev/null || true
    sed -i '' "s/return ET.tostring(data)/return ET.tostring(data, encoding='unicode')/g" src/dataset/serialize.py 2>/dev/null || true
    sed -i '' 's/range(min_level, max_level + 1)/range(int(min_level), int(max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
    sed -i '' 's/range(self.min_level, self.max_level + 1)/range(int(self.min_level), int(self.max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
else
    # Linux
    sed -i 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/AoT.py 2>/dev/null || true
    sed -i 's/from scipy.misc import comb/from scipy.special import comb/g' src/dataset/sampling.py 2>/dev/null || true
    sed -i "s/return ET.tostring(data)/return ET.tostring(data, encoding='unicode')/g" src/dataset/serialize.py 2>/dev/null || true
    sed -i 's/range(min_level, max_level + 1)/range(int(min_level), int(max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
    sed -i 's/range(self.min_level, self.max_level + 1)/range(int(self.min_level), int(self.max_level) + 1)/g' src/dataset/Attribute.py 2>/dev/null || true
fi

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
echo -e "\n${YELLOW}[4/5] Generating RAVEN dataset (this may take a few minutes)...${NC}"
mkdir -p data/raven_small

cd RAVEN
PYTHONPATH=src python -m dataset.main --num-samples 200 --save-dir ../data/raven_small
cd "$SCRIPT_DIR"

# Count generated files
NUM_FILES=$(find data/raven_small -name "*.npz" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Dataset generated: ${NUM_FILES} puzzle files${NC}"

# Step 5: Verify setup
echo -e "\n${YELLOW}[5/5] Verifying setup...${NC}"
python test_setup.py

echo -e "\n${GREEN}=============================================="
echo "SETUP COMPLETE!"
echo "==============================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Train models:         python train.py --epochs 15"
echo "  3. Evaluate:             python evaluate.py"
echo "  4. Run simulator:        streamlit run raven_simulator.py"
echo ""

