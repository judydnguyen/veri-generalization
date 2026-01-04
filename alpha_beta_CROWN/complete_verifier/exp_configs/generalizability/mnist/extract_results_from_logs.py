#!/usr/bin/env python3
"""
Script to extract verification results from log files and write to CSV.

Extracts:
- safe: number of verified safe instances
- unsafe: number of falsified (unsafe) instances
- unknown: number of unknown/timeout instances
- time: mean time for all instances
- epsilon: epsilon value from filename
- model path: model path from config
"""

import os
import re
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Default directory containing log files (can be overridden via command line)
LOGS_DIR = Path(__file__).parent / "logs"
OUTPUT_CSV = Path(__file__).parent / "results" / "extracted_results.csv"


def extract_epsilon_from_filename(filename):
    """Extract epsilon value from log filename."""
    # Pattern: eps_1over255, eps_2over255, etc.
    match = re.search(r'eps_(\d+)over255', filename)
    if match:
        num = int(match.group(1))
        return f"{num}/255"
    
    # Try other patterns
    match = re.search(r'eps[_-]?0?\.?(\d+)', filename)
    if match:
        return f"0.{match.group(1)}"
    
    return "N/A"


def extract_model_path(log_content):
    """Extract model path from log file content."""
    # Look for "path:" in the model section
    lines = log_content.split('\n')
    in_model_section = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('model:'):
            in_model_section = True
            continue
        
        if in_model_section:
            # Check if we've left the model section (next top-level key)
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                if ':' in line and not line.strip().startswith('#'):
                    break
            
            # Look for path: in model section
            match = re.search(r'^\s+path:\s+(.+)$', line)
            if match:
                return match.group(1).strip()
    
    return "N/A"


def extract_results_from_log(log_file):
    """
    Extract verification results from a log file.
    
    Returns:
        dict with keys: safe, unsafe, unknown, time, epsilon, model_path
    """
    results = {
        'safe': 0,
        'unsafe': 0,
        'unknown': 0,
        'time': 0.0,
        'epsilon': extract_epsilon_from_filename(log_file.name),
        'model_path': 'N/A'
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract model path
        results['model_path'] = extract_model_path(content)
        
        # Extract safe count
        safe_match = re.search(r'safe\s+\(total\s+(\d+)\)', content)
        if safe_match:
            results['safe'] = int(safe_match.group(1))
        
        # Extract unsafe count (could be "unsafe-pgd", "unsafe", etc.)
        unsafe_match = re.search(r'unsafe[-\w]*\s+\(total\s+(\d+)\)', content)
        if unsafe_match:
            results['unsafe'] = int(unsafe_match.group(1))
        
        # Extract summary line: "Problem instances count: X , total verified (safe/unsat): Y , total falsified (unsafe/sat): Z , timeout: W"
        summary_match = re.search(
            r'Problem instances count:\s*(\d+)\s*,\s*total verified.*?:\s*(\d+)\s*,\s*total falsified.*?:\s*(\d+)\s*,\s*timeout:\s*(\d+)',
            content
        )
        if summary_match:
            total = int(summary_match.group(1))
            verified = int(summary_match.group(2))
            falsified = int(summary_match.group(3))
            timeout = int(summary_match.group(4))
            
            # Recalculate to ensure consistency
            results['safe'] = verified
            results['unsafe'] = falsified
            results['unknown'] = timeout + (total - verified - falsified - timeout)
        
        # Extract mean time
        time_match = re.search(r'mean time for ALL instances\s*\(total\s+\d+\):\s*([\d.]+)', content)
        if time_match:
            results['time'] = float(time_match.group(1))
        
        # If we didn't find unknown from summary, try to calculate it
        if results['unknown'] == 0:
            # Try to find timeout count separately
            timeout_match = re.search(r'timeout\s+\(total\s+(\d+)\)', content)
            unknown_match = re.search(r'unknown\s+\(total\s+(\d+)\)', content)
            
            timeout_count = int(timeout_match.group(1)) if timeout_match else 0
            unknown_count = int(unknown_match.group(1)) if unknown_match else 0
            results['unknown'] = timeout_count + unknown_count
        
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return None
    
    return results


def main():
    """Main function to extract results from all log files."""
    global LOGS_DIR, OUTPUT_CSV
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract verification results from log files')
    parser.add_argument('--logs-dir', type=str, help='Directory containing log files')
    parser.add_argument('--output-csv', type=str, help='Output CSV file path')
    args = parser.parse_args()
    
    # Override defaults if provided
    if args.logs_dir:
        LOGS_DIR = Path(args.logs_dir)
    if args.output_csv:
        OUTPUT_CSV = Path(args.output_csv)
    
    if not LOGS_DIR.exists():
        print(f"ERROR: Logs directory not found: {LOGS_DIR}")
        return
    
    # Find all log files
    log_files = list(LOGS_DIR.glob("*.log"))
    
    if not log_files:
        print(f"WARNING: No log files found in {LOGS_DIR}")
        return
    
    print(f"Found {len(log_files)} log file(s)")
    
    # Extract results from each log file
    all_results = []
    for log_file in sorted(log_files):
        print(f"Processing: {log_file.name}")
        results = extract_results_from_log(log_file)
        if results:
            results['log_file'] = log_file.name
            all_results.append(results)
        else:
            print(f"  WARNING: Failed to extract results from {log_file.name}")
    
    if not all_results:
        print("ERROR: No results extracted from any log file")
        return
    
    # Create output directory if it doesn't exist
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        fieldnames = ['log_file', 'epsilon', 'safe', 'unsafe', 'unknown', 'time', 'model_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            # Write only the requested columns (excluding log_file from output)
            writer.writerow({
                'epsilon': result['epsilon'],
                'safe': result['safe'],
                'unsafe': result['unsafe'],
                'unknown': result['unknown'],
                'time': f"{result['time']:.4f}",
                'model_path': result['model_path']
            })
    
    print(f"\n{'='*80}")
    print("Results extracted successfully!")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("Summary:")
    print(f"{'Log File':<50} {'Epsilon':<12} {'Safe':<6} {'Unsafe':<7} {'Unknown':<8} {'Time (s)':<12}")
    print("-" * 100)
    for result in all_results:
        print(f"{result['log_file']:<50} {result['epsilon']:<12} {result['safe']:<6} "
              f"{result['unsafe']:<7} {result['unknown']:<8} {result['time']:<12.4f}")
    
    # Print totals
    total_safe = sum(r['safe'] for r in all_results)
    total_unsafe = sum(r['unsafe'] for r in all_results)
    total_unknown = sum(r['unknown'] for r in all_results)
    avg_time = sum(r['time'] for r in all_results) / len(all_results) if all_results else 0
    
    print("-" * 100)
    print(f"{'TOTALS/AVERAGE':<50} {'':<12} {total_safe:<6} {total_unsafe:<7} {total_unknown:<8} {avg_time:<12.4f}")


if __name__ == "__main__":
    main()

