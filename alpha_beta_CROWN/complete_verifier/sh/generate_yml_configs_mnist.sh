#!/bin/bash
# Script to generate YAML config files for EMNIST FC verification
# Usage: ./generate_yml_configs.sh [model_paths...] [--root_path PATH] [--eps_values "1 2 3 4"] [--output_dir PATH] [--model_name NAME]
# Example: ./generate_yml_configs.sh model1.pt model2.pt --root_path /path/to/emnist --eps_values "1 2 3 4"
# 
# Default paths:
#   - Models: All *.pt files in DEFAULT_MODEL_DIR (see below)
#   - Root: <CODE_DIR>/generate_properties/mnistfc/<DATASET>
#   - CSV files expected: emnistfc_instances_eps{1,2,3,4}over255.csv

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# CODE_DIR = repo root (veri-generalization/), 3 levels up from sh/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default values for mnist
DATASET="mnist"
# Default list of model paths to process (populated from DEFAULT_MODEL_DIR)
DEFAULT_MODEL_DIR="${CODE_DIR}/checkpoints/mnistfc"
DEFAULT_MODEL_PATHS=()
if [ -d "$DEFAULT_MODEL_DIR" ]; then
    while IFS= read -r -d '' f; do
        DEFAULT_MODEL_PATHS+=("$f")
    done < <(find "$DEFAULT_MODEL_DIR" -maxdepth 1 -name "*.pt" -print0 | sort -z)
fi
# Root path: CODE_DIR + generate_properties location for this model/dataset
DEFAULT_ROOT_PATH="${CODE_DIR}/generate_properties/mnistfc/${DATASET}"
DEFAULT_EPS_VALUES="1 2 3 4"
DEFAULT_OUTPUT_DIR="${CODE_DIR}/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/${DATASET}"
DEFAULT_MODEL_NAME="mnist_fc"

# Parse command line arguments
MODEL_PATHS=()
ROOT_PATH="$DEFAULT_ROOT_PATH"
EPS_VALUES="$DEFAULT_EPS_VALUES"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
MODEL_NAME="$DEFAULT_MODEL_NAME"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root_path)
            ROOT_PATH="$2"
            shift 2
            ;;
        --eps_values)
            EPS_VALUES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            # Assume it's a model path
            MODEL_PATHS+=("$1")
            shift
            ;;
    esac
done

# Use default model paths if none provided
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    MODEL_PATHS=("${DEFAULT_MODEL_PATHS[@]}")
fi

echo "================================================================================"
echo "Generating YAML Config Files for EMNIST FC Verification"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Number of Models: ${#MODEL_PATHS[@]}"
echo "  Model Paths:"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "    - $MODEL_PATH"
done
echo "  Root Path:  $ROOT_PATH"
echo "  Epsilon Values: $EPS_VALUES"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model Name: $MODEL_NAME"
echo ""

# Validate root path
if [ ! -d "$ROOT_PATH" ]; then
    echo -e "${YELLOW}WARNING: Root path directory not found: $ROOT_PATH${NC}"
    echo "  Will still generate configs, but verification may fail."
    echo ""
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate YAML configs for each model and epsilon value
TOTAL_GENERATED=0
MODEL_COUNT=0

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_COUNT=$((MODEL_COUNT + 1))
    
    # Validate model path
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${YELLOW}WARNING: Model file not found: $MODEL_PATH${NC}"
        echo "  Will still generate configs, but verification may fail."
        echo ""
    fi
    
    # Extract base name from model path for config filename
    MODEL_BASENAME=$(basename "$MODEL_PATH" .pt)
    
    echo "================================================================================"
    echo -e "${BLUE}Processing Model $MODEL_COUNT/${#MODEL_PATHS[@]}: $(basename "$MODEL_PATH")${NC}"
    echo "================================================================================"
    echo ""
    
    # Generate YAML config for each epsilon value
    GENERATED_COUNT=0
    for EPS in $EPS_VALUES; do
    # Format epsilon as "Xover255" for CSV filename (matches generate_emnist_properties.py)
    CSV_NAME="${DATASET}fc_instances_eps${EPS}over255.csv"
    
    # Check if CSV file exists
    CSV_PATH="${ROOT_PATH}/${CSV_NAME}"
    if [ ! -f "$CSV_PATH" ]; then
        echo -e "${YELLOW}WARNING: CSV file not found: $CSV_PATH${NC}"
        echo "  Will still generate config, but verification may fail."
    fi
    
    # Generate output filename
    # Extract epsilon info from model path if available, or use default
    if [[ "$MODEL_BASENAME" == *"0.03"* ]] || [[ "$MODEL_BASENAME" == *"0_03"* ]]; then
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_BASENAME}_eps_${EPS}over255.yaml"
    else
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_BASENAME}_eps_${EPS}over255.yaml"
    fi
    
    echo -e "${BLUE}Generating: $(basename "$OUTPUT_FILE")${NC}"
    
    # Generate YAML content
    cat > "$OUTPUT_FILE" << EOF
model:
  name: ${MODEL_NAME}
  # path: /home/judy/code/veri-generalization/checkpoints/mnistfc/mnist_fc.pt
  path: ${MODEL_PATH}
  input_shape: [-1, 1, 28, 28]

general:
  # The csv file contains a list of vnnlib specifications.
  root_path: ${ROOT_PATH}
  csv_name: ${CSV_NAME}
  enable_incomplete_verification: False
data:
  dataset: ${DATASET}
  start: 0
  end: 100
  std: [1.]
  mean: [0.]
attack:
  pgd_restarts: 50
solver:
  batch_size: 1024
  beta-crown:
    iteration: 20
bab:
  timeout: 60
EOF
    
        if [ -f "$OUTPUT_FILE" ]; then
            echo -e "  ${GREEN}✓ Created: $OUTPUT_FILE${NC}"
            GENERATED_COUNT=$((GENERATED_COUNT + 1))
            TOTAL_GENERATED=$((TOTAL_GENERATED + 1))
        else
            echo -e "  ${RED}✗ Failed to create: $OUTPUT_FILE${NC}"
        fi
        echo ""
    done
    
    echo -e "${GREEN}Generated $GENERATED_COUNT YAML config file(s) for $(basename "$MODEL_PATH")${NC}"
    echo ""
done

echo "================================================================================"
echo -e "${GREEN}Total: Generated $TOTAL_GENERATED YAML config file(s) for ${#MODEL_PATHS[@]} model(s)${NC}"
echo "================================================================================"
echo ""
echo "Generated files:"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    MODEL_BASENAME=$(basename "$MODEL_PATH" .pt)
    echo ""
    echo "  Model: $(basename "$MODEL_PATH")"
    for EPS in $EPS_VALUES; do
        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_BASENAME}_eps_${EPS}over255.yaml"
        if [ -f "$OUTPUT_FILE" ]; then
            echo "    - $OUTPUT_FILE"
        fi
    done
done
echo ""
echo "Next steps:"
echo "  1. Review the generated YAML files"
echo "  2. Run verification using: python abcrown.py --config <yaml_file>"
echo "  3. Or use run_verification.py to run all configs"
echo ""

