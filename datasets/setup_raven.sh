#!/bin/bash
# ==============================================
# RAVEN RPM Solver - Setup Script
# ==============================================
# This script sets up the entire project from scratch:
# 1. Creates virtual environment
# 2. Installs dependencies
# 3. Clones and patches RAVEN dataset generator
# 4. Applies parallel processing optimizations
# 5. Generates the dataset
# 6. Verifies the setup
#
# Usage:
#   ./setup.sh              # Default: generates medium dataset (2000 samples)
#   ./setup.sh small        # Small dataset (200 samples) - for quick testing
#   ./setup.sh medium       # Medium dataset (2000 samples) - recommended
#   ./setup.sh large        # Large dataset (5000 samples) - best performance
#   ./setup.sh large --xml  # Also generate XML files (not needed for training)
# ==============================================

set -e  # Exit on error

# Parse arguments
DATASET_SIZE="medium"
GENERATE_XML=0

for arg in "$@"; do
    case "$arg" in
        small|medium|large)
            DATASET_SIZE="$arg"
            ;;
        --xml)
            GENERATE_XML=1
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [small|medium|large] [--xml]"
            echo ""
            echo "Options:"
            echo "  small   - 200 samples per config (quick testing)"
            echo "  medium  - 2000 samples per config (recommended)"
            echo "  large   - 5000 samples per config (best performance)"
            echo "  --xml   - Also generate XML files (not needed for training)"
            exit 0
            ;;
    esac
done

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
        NUM_SAMPLES=5000
        DATASET_NAME="raven_large"
        ;;
esac

echo "=============================================="
echo "RAVEN RPM Solver - Setup Script"
echo "=============================================="
echo "Dataset size: $DATASET_SIZE ($NUM_SAMPLES samples per config)"
echo "Generate XML: $([ $GENERATE_XML -eq 1 ] && echo 'yes' || echo 'no')"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Activate virtual environment if it exists
echo -e "\n${YELLOW}[1/5] Checking virtual environment...${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}! No venv found. Using system Python.${NC}"
fi

# Step 2: Install dependencies
echo -e "\n${YELLOW}[2/6] Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Clone and patch RAVEN repository
echo -e "\n${YELLOW}[3/6] Setting up RAVEN dataset generator...${NC}"
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
            if re.match(r'\s*print\s+["\']', line) and 'print(' not in line:
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

# Fix absolute imports to relative imports
echo "  Fixing imports to relative imports..."
cat > /tmp/fix_imports.py << 'PYEOF'
import os
import re

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
            pattern = rf'^(from\s+)({module})(\s+import)'
            replacement = rf'\1.\2\3'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
            pattern2 = rf'^(import\s+)({module})(\s*$)'
            replacement2 = rf'from . \1\2\3'
            content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  Fixed imports: {filepath}")
    except Exception as e:
        print(f"  Error: {filepath}: {e}")

for file in os.listdir('src/dataset/'):
    if file.endswith('.py'):
        fix_imports(os.path.join('src/dataset/', file))
PYEOF
python /tmp/fix_imports.py 2>/dev/null || true

# Fix Python 3 range() not being a list
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

# Step 4: Create optimized main.py with parallel processing
echo -e "\n${YELLOW}[4/6] Applying optimizations (parallel processing, skip existing)...${NC}"

cat > RAVEN/src/dataset/main_optimized.py << 'PYEOF'
# -*- coding: utf-8 -*-

import argparse
import copy
import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm, trange

from .build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single)
from .const import IMAGE_SIZE, RULE_ATTR
from .rendering import render_panel
from .sampling import sample_attr, sample_attr_avail, sample_rules
from .serialize import dom_problem, serialize_aot, serialize_rules
from .solver import solve


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def generate_single_sample(task):
    """Generate a single sample - used for parallel processing."""
    k, key, root, save_dir, seed, val_prop, test_prop, generate_xml = task
    
    random.seed(seed + k)
    np.random.seed(seed + k)
    
    count_num = k % 10
    if count_num < (10 - val_prop - test_prop):
        set_name = "train"
    elif count_num < (10 - test_prop):
        set_name = "val"
    else:
        set_name = "test"
    
    npz_path = "{}/{}/RAVEN_{}_{}.npz".format(save_dir, key, k, set_name)
    if os.path.exists(npz_path):
        return None
    
    while True:
        rule_groups = sample_rules()
        new_root = root.prune(rule_groups)    
        if new_root is not None:
            break
    
    start_node = new_root.sample()

    row_1_1 = copy.deepcopy(start_node)
    for l in range(len(rule_groups)):
        rule_group = rule_groups[l]
        rule_num_pos = rule_group[0]
        row_1_2 = rule_num_pos.apply_rule(row_1_1)
        row_1_3 = rule_num_pos.apply_rule(row_1_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_1_2 = rule.apply_rule(row_1_1, row_1_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_1_3 = rule.apply_rule(row_1_2, row_1_3)
        if l == 0:
            to_merge = [row_1_1, row_1_2, row_1_3]
        else:
            merge_component(to_merge[1], row_1_2, l)
            merge_component(to_merge[2], row_1_3, l)
    row_1_1, row_1_2, row_1_3 = to_merge

    row_2_1 = copy.deepcopy(start_node)
    row_2_1.resample(True)
    for l in range(len(rule_groups)):
        rule_group = rule_groups[l]
        rule_num_pos = rule_group[0]
        row_2_2 = rule_num_pos.apply_rule(row_2_1)
        row_2_3 = rule_num_pos.apply_rule(row_2_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_2_2 = rule.apply_rule(row_2_1, row_2_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_2_3 = rule.apply_rule(row_2_2, row_2_3)
        if l == 0:
            to_merge = [row_2_1, row_2_2, row_2_3]
        else:
            merge_component(to_merge[1], row_2_2, l)
            merge_component(to_merge[2], row_2_3, l)
    row_2_1, row_2_2, row_2_3 = to_merge

    row_3_1 = copy.deepcopy(start_node)
    row_3_1.resample(True)
    for l in range(len(rule_groups)):
        rule_group = rule_groups[l]
        rule_num_pos = rule_group[0]
        row_3_2 = rule_num_pos.apply_rule(row_3_1)
        row_3_3 = rule_num_pos.apply_rule(row_3_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_3_2 = rule.apply_rule(row_3_1, row_3_2)
        for i in range(1, len(rule_group)):
            rule = rule_group[i]
            row_3_3 = rule.apply_rule(row_3_2, row_3_3)
        if l == 0:
            to_merge = [row_3_1, row_3_2, row_3_3]
        else:
            merge_component(to_merge[1], row_3_2, l)
            merge_component(to_merge[2], row_3_3, l)
    row_3_1, row_3_2, row_3_3 = to_merge

    imgs = [render_panel(row_1_1),
            render_panel(row_1_2),
            render_panel(row_1_3),
            render_panel(row_2_1),
            render_panel(row_2_2),
            render_panel(row_2_3),
            render_panel(row_3_1),
            render_panel(row_3_2),
            np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
    context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]
    modifiable_attr = sample_attr_avail(rule_groups, row_3_3)
    answer_AoT = copy.deepcopy(row_3_3)
    candidates = [answer_AoT]
    for j in range(7):
        component_idx, attr_name, min_level, max_level = sample_attr(modifiable_attr)
        answer_j = copy.deepcopy(answer_AoT)
        answer_j.sample_new(component_idx, attr_name, min_level, max_level, answer_AoT)
        candidates.append(answer_j)

    random.shuffle(candidates)
    answers = []
    for candidate in candidates:
        answers.append(render_panel(candidate))

    image = imgs[0:8] + answers
    target = candidates.index(answer_AoT)
    predicted = solve(rule_groups, context, candidates)
    meta_matrix, meta_target = serialize_rules(rule_groups)
    structure, meta_structure = serialize_aot(start_node)
    np.savez(npz_path, image=image, 
                       target=target, 
                       predict=predicted,
                       meta_matrix=meta_matrix,
                       meta_target=meta_target, 
                       structure=structure,
                       meta_structure=meta_structure)
    
    if generate_xml:
        xml_path = "{}/{}/RAVEN_{}_{}.xml".format(save_dir, key, k, set_name)
        with open(xml_path, "w") as f:
            dom = dom_problem(context + candidates, rule_groups)
            f.write(dom)
    
    return 1 if target == predicted else 0


def separate(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} parallel workers")

    for key in all_configs.keys():
        root = all_configs[key]
        
        tasks = []
        skipped = 0
        for k in range(args.num_samples):
            count_num = k % 10
            if count_num < (10 - args.val - args.test):
                set_name = "train"
            elif count_num < (10 - args.test):
                set_name = "val"
            else:
                set_name = "test"
            
            npz_path = "{}/{}/RAVEN_{}_{}.npz".format(args.save_dir, key, k, set_name)
            if os.path.exists(npz_path):
                skipped += 1
            else:
                tasks.append((k, key, root, args.save_dir, args.seed, args.val, args.test, args.xml))
        
        if skipped > 0:
            print(f"  {key}: Skipping {skipped} existing files, generating {len(tasks)} new samples")
        
        if len(tasks) == 0:
            print(f"  {key}: All samples already exist, skipping")
            continue
        
        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(generate_single_sample, tasks), total=len(tasks), desc=f"  {key}"))
        
        valid_results = [r for r in results if r is not None]
        acc = sum(valid_results)
        if valid_results:
            print("Accuracy of {}: {}".format(key, float(acc) / len(valid_results)))


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=20000,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="~/Datasets/",
                                 help="path to folder where the generated dataset will be saved.")
    main_arg_parser.add_argument("--seed", type=int, default=1234,
                                 help="random seed for dataset generation")
    main_arg_parser.add_argument("--fuse", type=int, default=0,
                                 help="whether to fuse different configurations")
    main_arg_parser.add_argument("--val", type=float, default=2,
                                 help="the proportion of the size of validation set")
    main_arg_parser.add_argument("--test", type=float, default=2,
                                 help="the proportion of the size of test set")
    main_arg_parser.add_argument("--xml", action="store_true", default=False,
                                 help="generate XML files (not needed for training)")
    args = main_arg_parser.parse_args()

    all_configs = {"center_single": build_center_single(),
                   "distribute_four": build_distribute_four(),
                   "distribute_nine": build_distribute_nine(),
                   "left_center_single_right_center_single": build_left_center_single_right_center_single(),
                   "up_center_single_down_center_single": build_up_center_single_down_center_single(),
                   "in_center_single_out_center_single": build_in_center_single_out_center_single(),
                   "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single()}

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.fuse:
        if not os.path.exists(os.path.join(args.save_dir, "fuse")):
            os.mkdir(os.path.join(args.save_dir, "fuse"))
    else:
        for key in all_configs.keys():
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        separate(args, all_configs)


if __name__ == "__main__":
    main()
PYEOF

echo -e "${GREEN}✓ Optimizations applied (parallel processing, skip existing files)${NC}"

# Step 5: Generate dataset
echo -e "\n${YELLOW}[5/6] Generating RAVEN dataset '$DATASET_NAME' (this may take a few minutes)...${NC}"
mkdir -p "$PARENT_DIR/data/$DATASET_NAME"

cd RAVEN

# Build the command with optional --xml flag
CMD="PYTHONPATH=src python -m dataset.main_optimized --num-samples $NUM_SAMPLES --save-dir $PARENT_DIR/data/$DATASET_NAME"
if [ $GENERATE_XML -eq 1 ]; then
    CMD="$CMD --xml"
fi

eval $CMD
cd "$SCRIPT_DIR"

# Count generated files
NUM_FILES=$(find "$PARENT_DIR/data/$DATASET_NAME" -name "*.npz" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ Dataset generated: ${NUM_FILES} puzzle files in data/$DATASET_NAME${NC}"

# Step 6: Verify setup
echo -e "\n${YELLOW}[6/6] Verifying setup...${NC}"
python test_setup.py

echo -e "\n${GREEN}=============================================="
echo "SETUP COMPLETE!"
echo "==============================================${NC}"
echo ""
echo "Dataset: $PARENT_DIR/data/$DATASET_NAME ($NUM_FILES puzzles)"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Train models:         python train.py --data_dir $PARENT_DIR/data/$DATASET_NAME --epochs 30"
echo "  3. Evaluate:             python evaluate.py --data_dir $PARENT_DIR/data/$DATASET_NAME"
echo "  4. Run simulator:        streamlit run raven_simulator.py"
echo ""
echo "Tip: For better model performance, use a larger dataset:"
echo "     ./setup.sh large   # Generates 35,000 puzzles"
echo ""
