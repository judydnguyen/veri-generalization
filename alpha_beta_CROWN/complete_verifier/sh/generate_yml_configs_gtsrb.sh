#!/bin/bash
# Script to generate YAML config files for GTSRB CNN verification
# Usage: ./generate_yml_configs_gtsrb.sh [model_paths...] [--root_path PATH] [--eps_values "1 2 3 4"] [--output_dir PATH] [--model_name NAME]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values for GTSRB
DATASET="gtsrb"

# Get project root (3 levels up from sh directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DEFAULT_MODEL_DIR="${PROJECT_ROOT}/checkpoints/gtsrb"
DEFAULT_MODEL_PATHS=()
if [ -d "$DEFAULT_MODEL_DIR" ]; then
    while IFS= read -r -d '' f; do
        DEFAULT_MODEL_PATHS+=("$f")
    done < <(find "$DEFAULT_MODEL_DIR" -maxdepth 1 -name "*.pt" -print0 | sort -z)
fi
# Root path should point to where generate_gtsrb_properties.py outputs files
DEFAULT_ROOT_PATH="${PROJECT_ROOT}/generate_properties/gtsrb/gtsrb"
DEFAULT_EPS_VALUES="1 2 3 4"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/gtsrb"
DEFAULT_MODEL_NAME="gtsrb_cnn"

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
echo "Generating YAML Config Files for GTSRB CNN Verification"
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
    # Format epsilon as "Xover255" for CSV filename
    CSV_NAME="gtsrb_instances_eps${EPS}over255.csv"

    # Check if CSV file exists
    CSV_PATH="${ROOT_PATH}/${CSV_NAME}"
    if [ ! -f "$CSV_PATH" ]; then
        echo -e "${YELLOW}WARNING: CSV file not found: $CSV_PATH${NC}"
        echo "  Will still generate config, but verification may fail."
    fi

    # Generate output filename
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL_BASENAME}_eps_${EPS}over255.yaml"

    echo -e "${BLUE}Generating: $(basename "$OUTPUT_FILE")${NC}"

    # Generate YAML content
    cat > "$OUTPUT_FILE" << EOF
model:
  name: ${MODEL_NAME}
  path: ${MODEL_PATH}
  input_shape: [-1, 3, 32, 32]

general:
  # The csv file contains a list of vnnlib specifications.
  root_path: ${ROOT_PATH}
  csv_name: ${CSV_NAME}
  enable_incomplete_verification: False

data:
  dataset: ${DATASET}
  start: 0
  end: 215
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
            echo -e "  ${GREEN}Created: $OUTPUT_FILE${NC}"
            GENERATED_COUNT=$((GENERATED_COUNT + 1))
            TOTAL_GENERATED=$((TOTAL_GENERATED + 1))
        else
            echo -e "  ${RED}Failed to create: $OUTPUT_FILE${NC}"
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
echo "  3. Or use run_all_configs.sh to run all configs"
echo ""

# USAGE (defaults use PROJECT_ROOT; run from repo root or pass paths):
#   bash generate_yml_configs_gtsrb.sh
# Or with explicit paths:
#   bash generate_yml_configs_gtsrb.sh --root_path "${PROJECT_ROOT}/generate_properties/gtsrb/gtsrb" \
#     --output_dir "${PROJECT_ROOT}/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/gtsrb" \
#     --eps_values "1 2 3 4" --model_name "gtsrb_cnn"
