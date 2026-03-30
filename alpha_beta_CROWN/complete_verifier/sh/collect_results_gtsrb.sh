#!/bin/bash
# Script to collect and extract verification results for GTSRB, BTSD, and GTSRB-OOD-PixMix
# Runs the extraction script once per dataset and writes a single combined CSV.
#
# Usage:
#   ./collect_results_gtsrb.sh [--logs-base DIR] [--output-dir DIR] [--output FILE] [--datasets LIST]
#
# Examples:
#   ./collect_results_gtsrb.sh
#   ./collect_results_gtsrb.sh --logs-base ../../expr03_logs
#   ./collect_results_gtsrb.sh --datasets "gtsrb btsd gtsrb_ood_pixmix" --output-dir ../../results/gtsrb --logs-base ../../expr03_logs
#   ./collect_results_gtsrb.sh --output ../../results/gtsrb_combined.csv

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Derive paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETE_VERIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

EXTRACT_SCRIPT="${COMPLETE_VERIFIER_DIR}/exp_configs.bak/generalizability/mnist/extract_results_from_logs.py"

# Defaults
LOGS_BASE="${COMPLETE_VERIFIER_DIR}/../expr03_logs"
OUTPUT_DIR="${COMPLETE_VERIFIER_DIR}/../results"
OUTPUT_FILE=""   # if set, overrides OUTPUT_DIR/combined name
DATASETS=("gtsrb" "btsd" "gtsrb_ood_pixmix")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --logs-base|-l)
            if [[ "$2" == /* ]]; then
                LOGS_BASE="$2"
            else
                LOGS_BASE="$(realpath -m "$SCRIPT_DIR/$2")"
            fi
            shift 2
            ;;
        --output-dir|-o)
            if [[ "$2" == /* ]]; then
                OUTPUT_DIR="$2"
            else
                OUTPUT_DIR="$(realpath -m "$SCRIPT_DIR/$2")"
            fi
            shift 2
            ;;
        --output)
            if [[ "$2" == /* ]]; then
                OUTPUT_FILE="$2"
            else
                OUTPUT_FILE="$(realpath -m "$SCRIPT_DIR/$2")"
            fi
            shift 2
            ;;
        --datasets|-d)
            IFS=' ' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--logs-base DIR] [--output-dir DIR] [--output FILE] [--datasets LIST]"
            echo ""
            echo "Options:"
            echo "  --logs-base, -l DIR    Base directory containing per-dataset log subdirs"
            echo "                         (default: <repo>/alpha_beta_CROWN/expr03_logs)"
            echo "  --output-dir, -o DIR   Directory for the combined output CSV"
            echo "                         (default: <repo>/alpha_beta_CROWN/results)"
            echo "  --output FILE          Full path for the combined output CSV"
            echo "                         (overrides --output-dir)"
            echo "  --datasets, -d LIST    Space-separated list of datasets to process"
            echo "                         (default: \"gtsrb btsd gtsrb_ood_pixmix\")"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --logs-base ../../expr03_logs"
            echo "  $0 --datasets \"gtsrb btsd\""
            echo "  $0 --output ../../results/gtsrb_combined.csv"
            exit 0
            ;;
        *)
            echo -e "${RED}ERROR: Unknown option: $1${NC}"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# Resolve combined output path
DATASET_NAMES=$(IFS='_'; echo "${DATASETS[*]}")
if [[ -n "$OUTPUT_FILE" ]]; then
    COMBINED_CSV="$OUTPUT_FILE"
else
    COMBINED_CSV="${OUTPUT_DIR}/${DATASET_NAMES}_results.csv"
fi

echo "================================================================================"
echo "Collecting GTSRB/BTSD Verification Results"
echo "================================================================================"
echo ""
echo "Logs base:   $LOGS_BASE"
echo "Output file: $COMBINED_CSV"
echo "Datasets:    ${DATASETS[*]}"
echo ""

# Check extract script
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo -e "${RED}ERROR: Extract script not found: $EXTRACT_SCRIPT${NC}"
    exit 1
fi

mkdir -p "$(dirname "$COMBINED_CSV")"

# Temp dir for per-dataset CSVs
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

TOTAL_DATASETS=${#DATASETS[@]}
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
COLLECTED_CSVS=()

for DATASET in "${DATASETS[@]}"; do
    LOGS_DIR="${LOGS_BASE}/${DATASET}"
    TMP_CSV="${TMPDIR_WORK}/${DATASET}_results.csv"

    echo "--------------------------------------------------------------------------------"
    echo -e "${BLUE}Dataset: $DATASET${NC}"
    echo "  Logs dir: $LOGS_DIR"
    echo ""

    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "${YELLOW}  WARNING: Logs directory not found: $LOGS_DIR — skipping.${NC}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    LOG_COUNT=$(find "$LOGS_DIR" -name "*.log" -type f 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -eq 0 ]; then
        echo -e "${YELLOW}  WARNING: No log files found in $LOGS_DIR — skipping.${NC}"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    echo "  Found $LOG_COUNT log file(s)"

    if python3 "$EXTRACT_SCRIPT" --logs-dir "$LOGS_DIR" --output-csv "$TMP_CSV"; then
        echo -e "  ${GREEN}✓ Extracted $DATASET${NC}"
        COLLECTED_CSVS+=("$TMP_CSV")
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "  ${RED}ERROR: Failed to extract results for $DATASET${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    echo ""
done

# Combine all per-dataset CSVs into one
if [ "${#COLLECTED_CSVS[@]}" -gt 0 ]; then
    echo "--------------------------------------------------------------------------------"
    echo "Combining ${#COLLECTED_CSVS[@]} dataset(s) into: $COMBINED_CSV"

    # Write header from first file, then data rows (skip header) from the rest
    head -n 1 "${COLLECTED_CSVS[0]}" > "$COMBINED_CSV"
    for CSV in "${COLLECTED_CSVS[@]}"; do
        tail -n +2 "$CSV" >> "$COMBINED_CSV"
    done

    echo ""
    echo "Combined CSV Contents:"
    echo "================================================================================"
    column -t -s',' "$COMBINED_CSV" 2>/dev/null || cat "$COMBINED_CSV"
    echo "================================================================================"
fi

echo ""
echo "================================================================================"
echo "Summary"
echo "================================================================================"
echo "  Datasets processed: $TOTAL_DATASETS"
echo -e "  ${GREEN}Successful: $SUCCESS_COUNT${NC}"
if [ "$SKIP_COUNT" -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped:    $SKIP_COUNT${NC}"
fi
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "  ${RED}Failed:     $FAIL_COUNT${NC}"
fi
if [ -f "$COMBINED_CSV" ]; then
    echo ""
    echo -e "  ${GREEN}Output: $COMBINED_CSV${NC}"
fi
echo "================================================================================"
echo "Done!"
echo "================================================================================"

[ "$FAIL_COUNT" -gt 0 ] && exit 1
exit 0
