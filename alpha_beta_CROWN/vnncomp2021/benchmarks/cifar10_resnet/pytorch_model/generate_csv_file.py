# Loop through all the vnnlib files in the directory and generate a csv file
# The csv file will have only the property name column
# The property name will be the relative path to the vnnlib file

import os
import csv
import argparse


def generate_csv_file(directory, output_csv=None, relative_to=None, recursive=True):
    """
    Generate a CSV file containing all .vnnlib files in the directory.
    
    Args:
        directory: Directory to search for .vnnlib files
        output_csv: Output CSV file path (default: properties.csv in the directory)
        relative_to: Base directory for relative paths (default: directory)
        recursive: Whether to search subdirectories recursively (default: True)
    
    Returns:
        Path to the generated CSV file
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist or is not a directory.")
    
    # Default output CSV file
    if output_csv is None:
        output_csv = os.path.join(directory, "properties.csv")
    
    # Default relative_to is the directory itself
    if relative_to is None:
        relative_to = directory
    else:
        relative_to = os.path.abspath(relative_to)
    
    directory = os.path.abspath(directory)
    
    # Find all .vnnlib files
    vnnlib_files = []
    
    if recursive:
        # Recursively search for all .vnnlib files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".vnnlib"):
                    file_path = os.path.join(root, file)
                    vnnlib_files.append(file_path)
    else:
        # Only search in the top-level directory
        for file in os.listdir(directory):
            if file.endswith(".vnnlib"):
                file_path = os.path.join(directory, file)
                vnnlib_files.append(file_path)
    
    # Sort files for consistent output
    vnnlib_files.sort()
    
    if len(vnnlib_files) == 0:
        print(f"Warning: No .vnnlib files found in directory '{directory}'")
        return output_csv
    
    # Convert to relative paths if relative_to is specified
    property_paths = []
    for file_path in vnnlib_files:
        if relative_to:
            try:
                # Get relative path from relative_to to file_path
                rel_path = os.path.relpath(file_path, relative_to)
            except ValueError:
                # If relative path can't be computed, use absolute path
                rel_path = file_path
        else:
            rel_path = file_path
        
        # Normalize path separators to forward slashes (for cross-platform compatibility)
        rel_path = rel_path.replace('\\', '/')
        property_paths.append([rel_path])
    
    # Write to CSV file
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write each property path as a single-column row
        writer.writerows(property_paths)
    
    print(f"✅ Generated CSV file with {len(property_paths)} properties: {output_csv}")
    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CSV file with all .vnnlib property files')
    parser.add_argument('--directory', type=str, required=True, help='Directory to search for .vnnlib files')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path (default: properties.csv in directory)')
    parser.add_argument('--relative_to', type=str, default=None, help='Base directory for relative paths (default: same as directory)')
    parser.add_argument('--no_recursive', action='store_true', help='Do not search subdirectories recursively')
    
    args = parser.parse_args()
    
    generate_csv_file(
        directory=args.directory,
        output_csv=args.output,
        relative_to=args.relative_to,
        recursive=not args.no_recursive
    )

# python generate_csv_file.py --directory /home/judy/code/unlearning-verification/alpha_beta_CROWN/vnncomp2021/benchmarks/cifar10_resnet/vnnlib_properties_pgd_filtered/eps_0.01