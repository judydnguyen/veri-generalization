#!/usr/bin/env python3
"""
Test script to plot selected BTSC and GTSRB images with semantic labels.

This script visualizes:
1. Selected BTSC images with their semantic labels and GTSRB mappings
2. Side-by-side comparison of BTSC and corresponding GTSRB images
"""

import os
import sys
import argparse

# Add current directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_btsd_properties import (
    plot_selected_images_with_labels,
    plot_btsc_gtsrb_comparison,
    find_btsd_csv,
    find_btsd_images_dir,
    BTSC_TO_GTSRB,
    BTSC_SEMANTIC,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot BTSC and GTSRB images with semantic labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--btsd-csv", 
        type=str, 
        default=None,
        help="Path to BTSC labels.csv file"
    )
    parser.add_argument(
        "--btsd-images", 
        type=str, 
        default=None,
        help="Path to BTSC images directory (Testing or Training)"
    )
    parser.add_argument(
        "--gtsrb-root", 
        type=str, 
        default=None,
        help="Path to GTSRB root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: script directory)"
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=2,
        help="Number of images to show per class"
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip BTSC vs GTSRB comparison plot"
    )

    args = parser.parse_args()

    # Base datasets directory
    datasets_base = "/home/judy/code/veri-generalization/datasets"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir if args.output_dir else script_dir

    # Resolve BTSC CSV
    if args.btsd_csv:
        btsd_csv = args.btsd_csv
    else:
        btsd_csv = os.path.join(datasets_base, "BTSC", "labels.csv")
        if not os.path.exists(btsd_csv):
            btsd_csv = os.path.join(datasets_base, "BTSD", "labels.csv")

    # Resolve BTSC images directory
    if args.btsd_images:
        btsd_images_dir = args.btsd_images
    else:
        btsd_images_dir = os.path.join(datasets_base, "BTSC", "Testing")
        if not os.path.isdir(btsd_images_dir):
            btsd_images_dir = os.path.join(datasets_base, "BTSC", "Training")

    # Resolve GTSRB root
    if args.gtsrb_root:
        gtsrb_root = args.gtsrb_root
    else:
        gtsrb_root = os.path.join(datasets_base, "GTSRB", "Train", "gtsrb", "GTSRB", "Training")
        if not os.path.isdir(gtsrb_root):
            gtsrb_root = os.path.join(datasets_base, "GTSRB", "Train")

    # Output file paths
    output_file = os.path.join(output_dir, "btsc_selected_images.png")
    comparison_file = os.path.join(output_dir, "btsc_gtsrb_comparison.png")

    print("=" * 60)
    print("BTSC and GTSRB Image Plotting")
    print("=" * 60)
    print(f"Datasets base directory: {datasets_base}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    try:
        # Robust path resolution
        print("\nResolving paths...")
        resolved_csv = find_btsd_csv(btsd_csv)
        resolved_images = find_btsd_images_dir(btsd_images_dir, resolved_csv)

        print("\n✓ Resolved paths:")
        print(f"  CSV file: {resolved_csv}")
        print(f"  BTSC images directory: {resolved_images}")
        print(f"  GTSRB root directory: {gtsrb_root}")
        print(f"\n  Output files:")
        print(f"    - BTSC only: {output_file}")
        if not args.skip_comparison:
            print(f"    - Comparison: {comparison_file}")
        print("=" * 60)

        # Print class mapping info
        print(f"\n📋 Class mappings ({len(BTSC_TO_GTSRB)} classes):")
        for btsc_id, gtsrb_label in sorted(BTSC_TO_GTSRB.items()):
            semantic = BTSC_SEMANTIC.get(btsc_id, "Unknown")
            print(f"  BTSC {btsc_id} → GTSRB {gtsrb_label}: {semantic}")
        print("=" * 60)

        # Plot BTSC images only
        print("\n📊 Plotting BTSC images...")
        plot_selected_images_with_labels(
            csv_file=resolved_csv,
            images_root=resolved_images,
            output_file=output_file,
            images_per_class=args.images_per_class
        )
        print(f"✅ BTSC plot saved to: {output_file}")

        # Plot BTSC vs GTSRB comparison
        if not args.skip_comparison:
            print("\n📊 Plotting BTSC vs GTSRB comparison...")
            plot_btsc_gtsrb_comparison(
                csv_file=resolved_csv,
                images_root=resolved_images,
                gtsrb_root=gtsrb_root,
                output_file=comparison_file,
                images_per_class=args.images_per_class,
                apply_perturbation=False
            )
            print(f"✅ Comparison plot saved to: {comparison_file}")

        print("\n" + "=" * 60)
        print("✅ All plots generated successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Make sure the datasets are downloaded and available.")
        print("   Expected locations:")
        print(f"     - BTSC CSV: {datasets_base}/BTSC/labels.csv")
        print(f"     - BTSC images: {datasets_base}/BTSC/Testing/ or Training/")
        print(f"     - GTSRB: {datasets_base}/GTSRB/Train/")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
