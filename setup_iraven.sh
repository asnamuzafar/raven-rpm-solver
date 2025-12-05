#!/bin/bash
# ==============================================
# I-RAVEN Dataset - Setup Script
# ==============================================
# This script sets up the I-RAVEN dataset (bias-corrected RAVEN):
# 1. Clones the i-raven repository
# 2. Installs dependencies
# 3. Generates the I-RAVEN dataset
#
# Usage:
#   ./setup_iraven.sh              # Default: generates medium dataset (2000 samples)
#   ./setup_iraven.sh small        # Small dataset (200 samples) - for quick testing
#   ./setup_iraven.sh medium       # Medium dataset (2000 samples) - recommended
#   ./setup_iraven.sh large        # Large dataset (5000 samples) - best performance
# ==============================================

set -e  # Exit on error

# Parse dataset size argument
DATASET_SIZE="${1:-medium}"  # Default to medium

case "$DATASET_SIZE" in
    small)
        NUM_SAMPLES=200
        DATASET_NAME="iraven_small"
        ;;
    medium)
        NUM_SAMPLES=2000
        DATASET_NAME="iraven_medium"
        ;;
    large)
        NUM_SAMPLES=5000
        DATASET_NAME="iraven_large"
        ;;
    *)
        echo "Unknown dataset size: $DATASET_SIZE"
        echo "Usage: ./setup_iraven.sh [small|medium|large]"
        exit 1
        ;;
esac

echo "=============================================="
echo "I-RAVEN Dataset - Setup Script"
echo "=============================================="
echo "Dataset size: $DATASET_SIZE ($NUM_SAMPLES samples per config)"
echo ""
echo "I-RAVEN is a bias-corrected version of RAVEN that"
echo "prevents shortcut learning in the answer choices."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Activate virtual environment if it exists
echo -e "\n${YELLOW}[1/4] Checking virtual environment...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}! No venv found. Using system Python.${NC}"
fi

# Step 2: Clone I-RAVEN repository
echo -e "\n${YELLOW}[2/4] Setting up I-RAVEN dataset generator...${NC}"
if [ ! -d "I-RAVEN" ]; then
    git clone https://github.com/cwhy/i-raven.git I-RAVEN
    echo -e "${GREEN}✓ I-RAVEN repository cloned${NC}"
else
    echo -e "${GREEN}✓ I-RAVEN repository already exists${NC}"
fi

# Step 3: Install I-RAVEN dependencies
echo -e "\n${YELLOW}[3/4] Installing I-RAVEN dependencies...${NC}"
cd I-RAVEN
pip install -r requirements.txt -q 2>/dev/null || pip install numpy scipy opencv-python pillow -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Generate I-RAVEN dataset
echo -e "\n${YELLOW}[4/4] Generating I-RAVEN dataset '$DATASET_NAME' (this may take a few minutes)...${NC}"
cd "$SCRIPT_DIR"
mkdir -p "data/$DATASET_NAME"

cd I-RAVEN
python main.py --num-samples "$NUM_SAMPLES" --save-dir "../data/$DATASET_NAME"
cd "$SCRIPT_DIR"

# Count generated files
NUM_FILES=$(find "data/$DATASET_NAME" -name "*.npz" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ I-RAVEN dataset generated: ${NUM_FILES} puzzle files in data/$DATASET_NAME${NC}"

echo -e "\n${GREEN}=============================================="
echo "I-RAVEN SETUP COMPLETE!"
echo "==============================================${NC}"
echo ""
echo "Dataset: data/$DATASET_NAME ($NUM_FILES puzzles)"
echo ""
echo "To use I-RAVEN for training:"
echo "  1. Edit config.py: DATASET_TYPE = 'iraven'"
echo "  2. Or use CLI:     python train.py --data_dir ./data/$DATASET_NAME"
echo ""
echo "To compare RAVEN vs I-RAVEN:"
echo "  python train.py --data_dir ./data/raven_medium --epochs 10"
echo "  python train.py --data_dir ./data/$DATASET_NAME --epochs 10"
echo ""
