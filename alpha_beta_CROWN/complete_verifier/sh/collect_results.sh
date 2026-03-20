#!/bin/bash
# Script to collect and extract verification results from log files
# Extracts results from verification logs and writes to CSV
#
# Usage:
#   ./collect_results.sh [logs_dir] [--output OUTPUT_FILE]
#   ./collect_results.sh --output OUTPUT_FILE
#
# Examples:
#   ./collect_results.sh ../../expr03_logs/mnist
#   ./collect_results.sh ../../expr03_logs/mnist --output ../../results/mnist_results.csv
#   ./collect_results.sh ../../expr03_logs/gtsrb --output ../../results/gtsrb_results.csv

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Derive paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETE_VERIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

EXTRACT_SCRIPT="${COMPLETE_VERIFIER_DIR}/exp_configs.bak/generalizability/mnist/extract_results_from_logs.py"

# Default paths (can be overridden via args)
LOGS_DIR="${CODE_DIR}/alpha_beta_CROWN/expr03_logs"
RESULTS_DIR="${CODE_DIR}/results"
OUTPUT_FILENAME="extracted_results.csv"
OUTPUT_CSV="${RESULTS_DIR}/${OUTPUT_FILENAME}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output|-o)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [logs_dir] [--output OUTPUT_FILE]"
            echo ""
            echo "Options:"
            echo "  logs_dir              Directory containing log files"
            echo "  --output, -o FILE     Output CSV filename (default: extracted_results.csv)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 ../expr03_logs/mnist"
            echo "  $0 ../expr03_logs/mnist --output ../../results/mnist_results.csv"
            echo "  $0 ../expr03_logs/gtsrb --output ../../results/gtsrb_results.csv"
            exit 0
            ;;
        --*)
            echo -e "${RED}ERROR: Unknown option: $1${NC}"
            echo "Did you mean --output?"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
        *)
            # Treat as logs directory path
            if [ -n "$1" ]; then
                # Resolve relative paths against SCRIPT_DIR, then normalize
                if [[ "$1" == /* ]]; then
                    _raw="$1"
                else
                    _raw="$SCRIPT_DIR/$1"
                fi
                LOGS_DIR="$(cd "$_raw" 2>/dev/null && pwd || echo "$_raw")"
                RESULTS_DIR="$(dirname "$LOGS_DIR")/results"
                echo -e "${GREEN}Using logs directory: $LOGS_DIR${NC}"
            fi
            shift
            ;;
    esac
done

# Set final output CSV path
# If OUTPUT_FILENAME contains any path separator, resolve it (absolute or relative to SCRIPT_DIR)
if [[ "$OUTPUT_FILENAME" == /* ]]; then
    OUTPUT_CSV="$OUTPUT_FILENAME"
    RESULTS_DIR="$(dirname "$OUTPUT_CSV")"
elif [[ "$OUTPUT_FILENAME" == */* ]]; then
    # Relative path with directories - resolve from SCRIPT_DIR (works even if dir doesn't exist)
    OUTPUT_CSV="$(realpath -m "$SCRIPT_DIR/$OUTPUT_FILENAME")"
    RESULTS_DIR="$(dirname "$OUTPUT_CSV")"
else
    OUTPUT_CSV="${RESULTS_DIR}/${OUTPUT_FILENAME}"
fi

echo "================================================================================"
echo "Collecting Verification Results from Log Files"
echo "================================================================================"
echo ""
echo "Logs directory: $LOGS_DIR"
echo "Output file: $OUTPUT_CSV"
echo ""

# Check if extract script exists
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    echo -e "${RED}ERROR: Extract script not found: $EXTRACT_SCRIPT${NC}"
    exit 1
fi

# Check if logs directory exists
if [ ! -d "$LOGS_DIR" ]; then
    echo -e "${YELLOW}WARNING: Logs directory not found: $LOGS_DIR${NC}"
    echo "Creating directory..."
    mkdir -p "$LOGS_DIR"
fi

# Ensure output path is not a directory (can happen from a previous buggy run)
if [ -d "$OUTPUT_CSV" ]; then
    echo -e "${RED}ERROR: Output path exists as a directory: $OUTPUT_CSV${NC}"
    echo "This was likely created by a previous run with a mistyped flag."
    echo "Remove it with:  rm -rf \"$OUTPUT_CSV\""
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Count log files
LOG_COUNT=$(find "$LOGS_DIR" -name "*.log" -type f 2>/dev/null | wc -l)

if [ "$LOG_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}WARNING: No log files found in $LOGS_DIR${NC}"
    echo "Please run verification first to generate log files."
    exit 1
fi

echo "Found $LOG_COUNT log file(s) in $LOGS_DIR"
echo ""

# Run the extraction script
echo "Running extraction script..."
echo ""

if python3 "$EXTRACT_SCRIPT" --logs-dir "$LOGS_DIR" --output-csv "$OUTPUT_CSV"; then
    echo ""
    echo -e "${GREEN}✓ Results extracted successfully!${NC}"
    echo ""

    # Check if output CSV was created
    if [ -f "$OUTPUT_CSV" ]; then
        echo "Output CSV: $OUTPUT_CSV"
        echo ""
        echo "CSV Contents:"
        echo "================================================================================"
        column -t -s',' "$OUTPUT_CSV" 2>/dev/null || cat "$OUTPUT_CSV"
        echo "================================================================================"
        echo ""
        echo -e "${GREEN}Results saved to: $OUTPUT_CSV${NC}"
    else
        echo -e "${YELLOW}WARNING: Output CSV not found at expected location: $OUTPUT_CSV${NC}"
    fi
else
    echo ""
    echo -e "${RED}ERROR: Failed to extract results${NC}"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Done!"
echo "================================================================================"
