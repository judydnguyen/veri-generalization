#!/bin/bash

#########################################################################
# Script to run abcrown.py on all .yaml configuration files            #
#                                                                       #
# Usage:                                                                #
#   ./run_all_configs.sh [OPTIONS]                                     #
#                                                                       #
# Options:                                                              #
#   --config-dir DIR       Directory to search for yaml files          #
#                          (default: ./exp_configs/generalizability)   #
#   --log-dir DIR          Directory to save log files                 #
#                          (default: ../logs)                          #
#   --pattern PATTERN      Only run configs matching pattern           #
#   --dry-run              Show what would be run without executing    #
#   --parallel N           Run N jobs in parallel (default: 1)         #
#   --continue-on-error    Continue even if a job fails                #
#   --skip-existing        Skip if log file already exists             #
#   --help                 Show this help message                      #
#########################################################################

# Don't exit on error - we handle errors ourselves
set +e

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETE_VERIFIER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$COMPLETE_VERIFIER_DIR/exp_configs/generalizability/mnist"
LOG_DIR="$COMPLETE_VERIFIER_DIR/../logs"
PATTERN=""
DRY_RUN=false
PARALLEL=1
CONTINUE_ON_ERROR=false
SKIP_EXISTING=false

# Default list of model prefixes to process
# These should match the basenames of model files (without .pt extension)
DEFAULT_MODEL_PREFIXES=(
    "mnist_fc_eps"
    "mnist_fc_adv_eps0.03_acc93.17"
    "mnist_fc_adv_eps0.01_acc93.60"
    "mnist_fc_adv_eps0.05_acc93.24"
)

MODEL_PREFIXES=("${DEFAULT_MODEL_PREFIXES[@]}")
USE_MODEL_PREFIXES=true

# Default list of model prefixes to process
# These should match the basenames of model files (without .pt extension)
DEFAULT_MODEL_PREFIXES=(
    "mnist_fc_eps"
    "mnist_fc_adv_eps0.03_acc93.17"
    "mnist_fc_adv_eps0.01_acc93.60"
    "mnist_fc_adv_eps0.05_acc93.24"
)

MODEL_PREFIXES=()
USE_MODEL_PREFIXES=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --pattern)
            PATTERN="$2"
            USE_MODEL_PREFIXES=false  # Pattern overrides prefix mode
            shift 2
            ;;
        --all-configs|--no-prefixes)
            USE_MODEL_PREFIXES=false
            shift
            ;;
        --model-prefixes|--prefixes)
            MODEL_PREFIXES=()
            USE_MODEL_PREFIXES=true
            # Parse space-separated prefixes
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODEL_PREFIXES+=("$1")
                shift
            done
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n 19
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to complete_verifier directory
cd "$COMPLETE_VERIFIER_DIR" || exit 1

# Resolve CONFIG_DIR to absolute path for existence check
CONFIG_DIR_ABS="$CONFIG_DIR"
if [[ "$CONFIG_DIR" != /* ]]; then
    CONFIG_DIR_ABS="$(cd "$CONFIG_DIR" 2>/dev/null && pwd)"
    [[ -z "$CONFIG_DIR_ABS" ]] && CONFIG_DIR_ABS="$COMPLETE_VERIFIER_DIR/$CONFIG_DIR"
fi
if [[ ! -d "$CONFIG_DIR_ABS" ]]; then
    echo -e "${RED}ERROR: Config directory does not exist: $CONFIG_DIR${NC}"
    echo "  Resolved to: $CONFIG_DIR_ABS"
    echo "  For this repo, use e.g.: --config-dir $COMPLETE_VERIFIER_DIR/exp_configs/generalizability/gtsrb"
    exit 1
fi
# Use resolved path so find always has an absolute path
CONFIG_DIR="$CONFIG_DIR_ABS"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running abcrown.py on all .yaml configs${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Config directory: ${GREEN}$CONFIG_DIR${NC}"
echo -e "Log directory:    ${GREEN}$LOG_DIR${NC}"
if [[ "$USE_MODEL_PREFIXES" == true ]]; then
    echo -e "Model prefixes:   ${GREEN}${#MODEL_PREFIXES[@]} prefix(es)${NC}"
    for prefix in "${MODEL_PREFIXES[@]}"; do
        echo -e "                   - $prefix"
    done
elif [[ -n "$PATTERN" ]]; then
    echo -e "Pattern filter:   ${GREEN}${PATTERN}${NC}"
else
    echo -e "Filter:           ${GREEN}none (all configs)${NC}"
fi
echo -e "Parallel jobs:    ${GREEN}$PARALLEL${NC}"
echo -e "Dry run:          ${GREEN}$DRY_RUN${NC}"
echo -e "Continue on error: ${GREEN}$CONTINUE_ON_ERROR${NC}"
echo -e "Skip existing:    ${GREEN}$SKIP_EXISTING${NC}"
echo ""

# Function to find YAML files based on model prefixes or pattern
find_yaml_files() {
    local prefix="$1"
    if [[ -n "$prefix" ]]; then
        # Find YAML files matching this prefix
        find "$CONFIG_DIR" -type f -name "${prefix}_*.yaml" | sort
    elif [[ -n "$PATTERN" ]]; then
        find "$CONFIG_DIR" -type f -name "*.yaml" | grep -E "$PATTERN" | sort
    else
        find "$CONFIG_DIR" -type f -name "*.yaml" | sort
    fi
}

# Count total files for summary
TOTAL=0
if [[ "$USE_MODEL_PREFIXES" == true ]]; then
    for prefix in "${MODEL_PREFIXES[@]}"; do
        prefix_files=$(find_yaml_files "$prefix")
        if [[ -n "$prefix_files" ]]; then
            prefix_count=$(echo "$prefix_files" | wc -l)
            TOTAL=$((TOTAL + prefix_count))
        fi
    done
else
    YAML_FILES=$(find_yaml_files "")
    if [[ -z "$YAML_FILES" ]]; then
        echo -e "${RED}No .yaml files found in $CONFIG_DIR${NC}"
        exit 1
    fi
    TOTAL=$(echo "$YAML_FILES" | wc -l)
fi

if [[ $TOTAL -eq 0 ]]; then
    echo -e "${RED}No .yaml files found${NC}"
    if [[ "$USE_MODEL_PREFIXES" == true ]]; then
        echo "  Checked prefixes: ${MODEL_PREFIXES[*]}"
    fi
    exit 1
fi

echo -e "${BLUE}Found $TOTAL .yaml file(s) total${NC}"
echo ""

# Track results
TOTAL_SUCCESS=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

# Function to get relative path from one directory to a file
# Returns relative path if possible, otherwise returns absolute path
get_relative_path() {
    local from_dir="$1"
    local to_file="$2"
    
    # Normalize paths
    from_dir="$(cd "$from_dir" 2>/dev/null && pwd)"
    local to_dir="$(cd "$(dirname "$to_file")" 2>/dev/null && pwd)"
    local file_name="$(basename "$to_file")"
    
    if [[ -z "$from_dir" ]] || [[ -z "$to_dir" ]]; then
        # If we can't normalize, return the original path
        echo "$to_file"
        return
    fi
    
    # Check if to_dir is within from_dir (or equal)
    if [[ "$to_dir" == "$from_dir" ]] || [[ "$to_dir" == "$from_dir"/* ]]; then
        # Get relative path (handle to_dir == from_dir so strip yields empty)
        local rel_dir="${to_dir#$from_dir/}"
        [[ "$rel_dir" == "$to_dir" ]] && rel_dir=""  # strip failed, same dir
        if [[ -z "$rel_dir" ]]; then
            echo "./$file_name"
        else
            echo "./$rel_dir/$file_name"
        fi
    else
        # File is outside from_dir, use absolute path
        echo "$to_dir/$file_name"
    fi
}

# Function to process a single config
process_config() {
    local yaml_file="$1"
    
    # Normalize CONFIG_DIR to absolute path for path calculations
    local config_dir_abs
    if [[ "$CONFIG_DIR" == /* ]]; then
        # Already absolute
        config_dir_abs="$CONFIG_DIR"
    else
        # Relative path - resolve relative to complete_verifier_dir
        config_dir_abs="$(cd "$COMPLETE_VERIFIER_DIR" && cd "$CONFIG_DIR" 2>/dev/null && pwd)"
        if [[ -z "$config_dir_abs" ]]; then
            # Fallback: construct path manually
            config_dir_abs="$COMPLETE_VERIFIER_DIR/$CONFIG_DIR"
        fi
    fi
    config_dir_abs="${config_dir_abs%/}"  # Remove trailing slash
    
    # Normalize yaml_file to absolute path
    local yaml_file_abs
    if [[ "$yaml_file" == /* ]]; then
        # Already absolute
        yaml_file_abs="$yaml_file"
    else
        # Relative path - resolve it
        if [[ -f "$yaml_file" ]]; then
            yaml_file_abs="$(cd "$(dirname "$yaml_file")" 2>/dev/null && pwd)/$(basename "$yaml_file")"
        else
            # File doesn't exist or can't resolve - use as is
            yaml_file_abs="$yaml_file"
        fi
    fi
    
    # Get relative path from config_dir for display/log naming
    local rel_path="${yaml_file_abs#$config_dir_abs/}"
    # If substitution didn't work (paths don't match), try to get basename
    if [[ "$rel_path" == "$yaml_file_abs" ]]; then
        # File is not in config_dir - try to get relative path from config_dir
        local temp_path
        if [[ -f "$yaml_file_abs" ]]; then
            temp_path="$(python3 -c "import os; print(os.path.relpath('$yaml_file_abs', '$config_dir_abs'))" 2>/dev/null)"
            if [[ -n "$temp_path" ]] && [[ "$temp_path" != "$yaml_file_abs" ]]; then
                rel_path="$temp_path"
            else
                rel_path="$(basename "$yaml_file_abs")"
            fi
        else
            rel_path="$(basename "$yaml_file_abs")"
        fi
    fi
    
    # Get config path relative to complete_verifier_dir (current working directory)
    # abcrown.py can handle both relative and absolute paths
    local config_rel_path
    if [[ -f "$yaml_file_abs" ]]; then
        # Try to get relative path from complete_verifier_dir
        config_rel_path="$(get_relative_path "$COMPLETE_VERIFIER_DIR" "$yaml_file_abs")"
    else
        # File doesn't exist, use the path as provided
        config_rel_path="$yaml_file_abs"
    fi
    
    # Create log file name by replacing slashes with underscores
    # Use rel_path for cleaner log names (relative to config_dir)
    local log_name=$(echo "$rel_path" | sed 's|/|_|g' | sed 's|\.yaml$|.log|')
    local log_file="$LOG_DIR/$log_name"
    
    # Skip if log exists and --skip-existing is set
    if [[ "$SKIP_EXISTING" == true ]] && [[ -f "$log_file" ]]; then
        echo -e "${YELLOW}[SKIP]${NC} $rel_path (log exists: $log_name)"
        return 1  # Return 1 for skipped
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${BLUE}[DRY-RUN]${NC} python abcrown.py --config $config_rel_path > $log_file 2>&1"
        return 0  # Return 0 for success (dry-run)
    fi
    
    echo -e "${BLUE}[RUN]${NC} $rel_path -> $log_name"
    
    # Run abcrown.py - it accepts both relative and absolute paths
    if python abcrown.py --config "$config_rel_path" > "$log_file" 2>&1; then
        echo -e "${GREEN}[OK]${NC} $rel_path"
        return 0  # Return 0 for success
    else
        local exit_code=$?
        echo -e "${RED}[FAIL]${NC} $rel_path (exit code: $exit_code)"
        return 2  # Return 2 for failed
    fi
}

# Process all configs - loop through model prefixes
if [[ "$USE_MODEL_PREFIXES" == true ]]; then
    # Process by model prefix groups
    echo -e "${BLUE}Processing configs by model prefix${NC}"
    echo ""
    
    PREFIX_COUNT=0
    for prefix in "${MODEL_PREFIXES[@]}"; do
        PREFIX_COUNT=$((PREFIX_COUNT + 1))
        echo "================================================================================"
        echo -e "${BLUE}Processing Model Prefix $PREFIX_COUNT/${#MODEL_PREFIXES[@]}: $prefix${NC}"
        echo "================================================================================"
        echo ""
        
        # Find YAML files for this prefix
        prefix_files=$(find_yaml_files "$prefix")
        if [[ -z "$prefix_files" ]]; then
            echo -e "${YELLOW}No configs found for prefix: $prefix${NC}"
            echo ""
            continue
        fi
        
        prefix_total=$(echo "$prefix_files" | wc -l)
        echo -e "${BLUE}Found $prefix_total config file(s) for prefix: $prefix${NC}"
        echo ""
        
        # Track results for this prefix
        PREFIX_SUCCESS=0
        PREFIX_FAILED=0
        PREFIX_SKIPPED=0
        
        # Process files for this prefix
        while IFS= read -r yaml_file; do
            [[ -z "$yaml_file" ]] && continue
            
            process_config "$yaml_file"
            result=$?
            
            case "$result" in
                0)  # Success
                    ((PREFIX_SUCCESS++))
                    ((TOTAL_SUCCESS++))
                    ;;
                1)  # Skipped
                    ((PREFIX_SKIPPED++))
                    ((TOTAL_SKIPPED++))
                    ;;
                2)  # Failed
                    ((PREFIX_FAILED++))
                    ((TOTAL_FAILED++))
                    if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                        echo -e "${RED}Stopping due to error (use --continue-on-error to continue)${NC}"
                        break 2  # Break out of both loops
                    fi
                    ;;
            esac
            echo ""
        done <<< "$prefix_files"
        
        # Summary for this prefix
        echo -e "${GREEN}Prefix '$prefix': $PREFIX_SUCCESS successful, $PREFIX_SKIPPED skipped, $PREFIX_FAILED failed${NC}"
        echo ""
    done
else
    # Process all files at once (when pattern is used or --all-configs)
    echo -e "${BLUE}Running configs${NC}"
    echo ""
    
    if [[ $PARALLEL -gt 1 ]]; then
        echo -e "${YELLOW}Note: Parallel execution is not yet fully implemented. Running sequentially.${NC}"
        echo ""
    fi
    
    # Sequential execution
    while IFS= read -r yaml_file; do
        [[ -z "$yaml_file" ]] && continue
        
        process_config "$yaml_file"
        result=$?
        
        case "$result" in
            0)  # Success
                ((TOTAL_SUCCESS++))
                ;;
            1)  # Skipped
                ((TOTAL_SKIPPED++))
                ;;
            2)  # Failed
                ((TOTAL_FAILED++))
                if [[ "$CONTINUE_ON_ERROR" != true ]]; then
                    echo -e "${RED}Stopping due to error (use --continue-on-error to continue)${NC}"
                    break
                fi
                ;;
        esac
        echo ""
    done <<< "$YAML_FILES"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total configs:  ${BLUE}$TOTAL${NC}"
if [[ "$USE_MODEL_PREFIXES" == true ]]; then
    echo -e "Model prefixes: ${BLUE}${#MODEL_PREFIXES[@]}${NC}"
fi
if [[ $TOTAL_SKIPPED -gt 0 ]]; then
    echo -e "Skipped:        ${YELLOW}$TOTAL_SKIPPED${NC}"
fi
echo -e "Successful:     ${GREEN}$TOTAL_SUCCESS${NC}"
if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo -e "Failed:         ${RED}$TOTAL_FAILED${NC}"
fi
echo -e "${BLUE}========================================${NC}"

# Exit with error if any failed (unless continue-on-error is set)
if [[ $TOTAL_FAILED -gt 0 ]] && [[ "$CONTINUE_ON_ERROR" != true ]]; then
    exit 1
fi

exit 0


# Examples (run from alpha_beta_CROWN/complete_verifier/sh):
#   ./run_all_configs.sh --pattern "mnist_fc_eps" --continue-on-error
#   ./run_all_configs.sh --pattern "gtsrb_cnn_eps" --continue-on-error --config-dir ../exp_configs/generalizability/gtsrb
# Or with absolute path (must exist):
#   ./run_all_configs.sh --pattern "gtsrb_cnn_eps" --continue-on-error --config-dir /home/judy/code/veri-generalization/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/gtsrb

# ./run_all_configs.sh --pattern "gtsrb_cnn_eps" --continue-on-error --config-dir /home/judy/code/veri-generalization/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/gtsrb --log-dir /home/judy/code/veri-generalization/alpha_beta_CROWN/expr03_logs/gtsrb
# ./run_all_configs.sh --pattern "gtsrb_cnn_adv" --continue-on-error --config-dir /home/judy/code/veri-generalization/alpha_beta_CROWN/complete_verifier/exp_configs/generalizability/gtsrb --log-dir /home/judy/code/veri-generalization/alpha_beta_CROWN/expr03_logs/gtsrb