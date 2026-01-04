#!/bin/bash
# Script to collect and extract verification results from log files
# Extracts results from MNIST FC verification logs and writes to CSV
#
# Usage:
#   ./collect_results.sh [logs_dir] [--output OUTPUT_FILE]
#   ./collect_results.sh expr01_logs [--output OUTPUT_FILE]
#   ./collect_results.sh --output OUTPUT_FILE
#
# Examples:
#   ./collect_results.sh expr01_logs
#   ./collect_results.sh expr01_logs --output my_results.csv
#   ./collect_results.sh /path/to/logs --output custom_results.csv

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="/home/judy/code/unlearning-verification/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/mnist"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_results_from_logs.py"

# Default paths (can be overridden)
LOGS_DIR="${SCRIPT_DIR}/logs"
RESULTS_DIR="${SCRIPT_DIR}/results"
OUTPUT_FILENAME="extracted_results.csv"
OUTPUT_CSV="${RESULTS_DIR}/${OUTPUT_FILENAME}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output|-o)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        expr01_logs|--expr01)
            LOGS_DIR="/home/judy/code/unlearning-verification/alpha_beta_CROWN/expr01_logs"
            RESULTS_DIR="/home/judy/code/unlearning-verification/alpha_beta_CROWN/expr01_logs/results"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [logs_dir] [--output OUTPUT_FILE]"
            echo ""
            echo "Options:"
            echo "  logs_dir              Directory containing log files (or 'expr01_logs' for shortcut)"
            echo "  --output, -o FILE     Output CSV filename (default: extracted_results.csv)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 expr01_logs"
            echo "  $0 expr01_logs --output my_results.csv"
            echo "  $0 /path/to/logs --output custom_results.csv"
            exit 0
            ;;
        *)
            # Treat as logs directory path
            if [ -n "$1" ]; then
                LOGS_DIR="$1"
                RESULTS_DIR="$(dirname "$LOGS_DIR")/results"
                echo -e "${GREEN}Using custom logs directory: $LOGS_DIR${NC}"
            fi
            shift
            ;;
    esac
done

# Set final output CSV path
# If OUTPUT_FILENAME is an absolute path, use it directly; otherwise use RESULTS_DIR
if [[ "$OUTPUT_FILENAME" == /* ]]; then
    OUTPUT_CSV="$OUTPUT_FILENAME"
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


#  ./collect_results.sh /home/judy/code/unlearning-verification/alpha_beta_CROWN/expr01_logs/mnist --output /home/judy/code/unlearning-verification/results/final_results_mnist_fc.csv
# ./collect_results.sh expr01_logs --output /home/judy/code/unlearning-verification/results/final_results_mnist_fc.csv
