#!/bin/bash
# ==============================================
# I-RAVEN Dataset - Setup Script
# ==============================================
# This script sets up the I-RAVEN dataset (bias-corrected RAVEN):
# 1. Clones the i-raven repository
# 2. Installs dependencies
# 3. Patches for parallel processing & skip existing files
# 4. Generates the I-RAVEN dataset
#
# Usage:
#   ./setup_iraven.sh              # Default: generates medium dataset (2000 samples)
#   ./setup_iraven.sh small        # Small dataset (200 samples) - for quick testing
#   ./setup_iraven.sh medium       # Medium dataset (2000 samples) - recommended
#   ./setup_iraven.sh large        # Large dataset (5000 samples) - best performance
#   ./setup_iraven.sh large --xml  # Also generate XML files (not needed for training)
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
            echo "Usage: ./setup_iraven.sh [small|medium|large] [--xml]"
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
esac

echo "=============================================="
echo "I-RAVEN Dataset - Setup Script"
echo "=============================================="
echo "Dataset size: $DATASET_SIZE ($NUM_SAMPLES samples per config)"
echo "Generate XML: $([ $GENERATE_XML -eq 1 ] && echo 'yes' || echo 'no')"
echo ""
echo "I-RAVEN is a bias-corrected version of RAVEN that"
echo "prevents shortcut learning in the answer choices."

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

# Step 2: Clone I-RAVEN repository
echo -e "\n${YELLOW}[2/5] Setting up I-RAVEN dataset generator...${NC}"
if [ ! -d "I-RAVEN" ]; then
    git clone https://github.com/cwhy/i-raven.git I-RAVEN
    echo -e "${GREEN}✓ I-RAVEN repository cloned${NC}"
else
    echo -e "${GREEN}✓ I-RAVEN repository already exists${NC}"
fi

# Step 3: Install I-RAVEN dependencies
echo -e "\n${YELLOW}[3/5] Installing I-RAVEN dependencies...${NC}"
cd I-RAVEN
pip install -r requirements.txt -q 2>/dev/null || pip install numpy scipy opencv-python pillow tqdm -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Patch main.py for parallel processing & skip existing files
echo -e "\n${YELLOW}[4/5] Applying optimizations (parallel processing, skip existing)...${NC}"
cd "$SCRIPT_DIR"

# Create the optimized main.py
cat > I-RAVEN/main_optimized.py << 'PYEOF'
import argparse
import copy
import os
import random
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm, trange

from build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single)
from const import IMAGE_SIZE
from rendering import render_panel
from sampling import sample_attr_avail, sample_rules
from serialize import dom_problem, serialize_aot, serialize_rules
from solver import solve

# Global flag for XML generation
GENERATE_XML = False


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def generate_single_sample(task):
    """Generate a single I-RAVEN sample - used for parallel processing."""
    k, key, root, save_dir, seed, val_prop, test_prop, generate_xml = task
    
    # Set seed based on k for reproducibility
    random.seed(seed + k)
    np.random.seed(seed + k)
    
    count_num = k % 10
    if count_num < (10 - val_prop - test_prop):
        set_name = "train"
    elif count_num < (10 - test_prop):
        set_name = "val"
    else:
        set_name = "test"
    
    # Skip if file already exists
    npz_path = "{}/{}/RAVEN_{}_{}.npz".format(save_dir, key, k, set_name)
    if os.path.exists(npz_path):
        return None  # Skip existing
    
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

    attr_num = 3
    if attr_num <= len(modifiable_attr):
        idx = np.random.choice(len(modifiable_attr), attr_num, replace=False)
        selected_attr = [modifiable_attr[i] for i in idx]
    else:
        selected_attr = modifiable_attr

    mode = None
    pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Number']
    if pos:
        pos = pos[0]
        selected_attr[pos], selected_attr[-1] = selected_attr[-1], selected_attr[pos]
        pos = [i for i in range(len(selected_attr)) if selected_attr[i][1] == 'Position']
        if pos:
            mode = 'Position-Number'
    values = []
    if len(selected_attr) >= 3:
        mode_3 = None
        if mode == 'Position-Number':
            mode_3 = '3-Position-Number'
        for i in range(attr_num):
            component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[i][0], selected_attr[i][1], \
                                                                       selected_attr[i][3], selected_attr[i][4], \
                                                                       selected_attr[i][5]
            value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, mode_3)
            values.append(value)
            tmp = []
            for j in candidates:
                new_AoT = copy.deepcopy(j)
                new_AoT.apply_new_value(component_idx, attr_name, value)
                tmp.append(new_AoT)
            candidates += tmp

    elif len(selected_attr) == 2:
        component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], \
                                                                   selected_attr[0][3], selected_attr[0][4], \
                                                                   selected_attr[0][5]
        value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
        values.append(value)
        new_AoT = copy.deepcopy(answer_AoT)
        new_AoT.apply_new_value(component_idx, attr_name, value)
        candidates.append(new_AoT)
        component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[1][0], selected_attr[1][1], \
                                                                   selected_attr[1][3], selected_attr[1][4], \
                                                                   selected_attr[1][5]
        if mode == 'Position-Number':
            ran, qu = 6, 1
        else:
            ran, qu = 3, 2
        for i in range(ran):
            value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
            values.append(value)
            for j in range(qu):
                new_AoT = copy.deepcopy(candidates[j])
                new_AoT.apply_new_value(component_idx, attr_name, value)
                candidates.append(new_AoT)

    elif len(selected_attr) == 1:
        component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], \
                                                                   selected_attr[0][3], selected_attr[0][4], \
                                                                   selected_attr[0][5]
        for i in range(7):
            value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
            values.append(value)
            new_AoT = copy.deepcopy(answer_AoT)
            new_AoT.apply_new_value(component_idx, attr_name, value)
            candidates.append(new_AoT)

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
    
    # Optionally generate XML
    if generate_xml:
        xml_path = "{}/{}/RAVEN_{}_{}.xml".format(save_dir, key, k, set_name)
        with open(xml_path, "wb") as f:
            dom = dom_problem(context + candidates, rule_groups)
            f.write(dom)

    return 1 if target == predicted else 0


def separate(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    num_workers = min(cpu_count(), 8)
    print(f"Using {num_workers} parallel workers")

    for key in list(all_configs.keys()):
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
    main_arg_parser = argparse.ArgumentParser(description="parser for I-RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=10000,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="/media/dsg3/datasets/I-RAVEN",
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
    if not args.fuse:
        for key in list(all_configs.keys()):
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        separate(args, all_configs)


if __name__ == "__main__":
    main()
PYEOF

echo -e "${GREEN}✓ Optimizations applied (parallel processing, skip existing files)${NC}"

# Step 5: Generate I-RAVEN dataset
echo -e "\n${YELLOW}[5/5] Generating I-RAVEN dataset '$DATASET_NAME' (this may take a few minutes)...${NC}"
cd "$SCRIPT_DIR"
mkdir -p "$PARENT_DIR/data/$DATASET_NAME"

cd I-RAVEN

# Build the command with optional --xml flag
CMD="python main_optimized.py --num-samples $NUM_SAMPLES --save-dir $PARENT_DIR/data/$DATASET_NAME"
if [ $GENERATE_XML -eq 1 ]; then
    CMD="$CMD --xml"
fi

eval $CMD
cd "$SCRIPT_DIR"

# Count generated files
NUM_FILES=$(find "$PARENT_DIR/data/$DATASET_NAME" -name "*.npz" | wc -l | tr -d ' ')
echo -e "${GREEN}✓ I-RAVEN dataset generated: ${NUM_FILES} puzzle files in $PARENT_DIR/data/$DATASET_NAME${NC}"

echo -e "\n${GREEN}=============================================="
echo "I-RAVEN SETUP COMPLETE!"
echo "==============================================${NC}"
echo ""
echo "Dataset: $PARENT_DIR/data/$DATASET_NAME ($NUM_FILES puzzles)"
echo ""
echo "To use I-RAVEN for training:"
echo "  1. Edit config.py: DATASET_TYPE = 'iraven'"
echo "  2. Or use CLI:     python train.py --data_dir $PARENT_DIR/data/$DATASET_NAME"
echo ""
echo "To compare RAVEN vs I-RAVEN:"
echo "  python train.py --data_dir $PARENT_DIR/data/raven_large --epochs 10"
echo "  python train.py --data_dir $PARENT_DIR/data/$DATASET_NAME --epochs 10"
echo ""
