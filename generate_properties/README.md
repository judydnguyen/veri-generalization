# Property Generation Scripts

This directory contains scripts for generating VNNLIB property files for neural network verification. The scripts generate adversarial robustness properties for MNIST, EMNIST, and PixMix OOD datasets.

## Overview

The property generation scripts create VNNLIB format files that specify input-output constraints for neural network verification. Each property defines:
- **Input constraints**: Bounds on input pixels (image ± epsilon perturbation)
- **Output constraints**: Classification robustness (correct class must have highest logit)

## Directory Structure

```
generate_properties/
├── README.md
└── mnistfc/
    ├── generate_properties.py          # Basic MNIST property generator
    ├── generate_mnist_properties.py    # Enhanced MNIST generator (balanced per-class)
    ├── generate_emnist_properties.py   # EMNIST property generator
    ├── generate_pixmix_mnist_properties.py  # PixMix OOD property generator
    ├── mnist-net_256x2.onnx            # ONNX models for verification
    ├── mnist-net_256x4.onnx
    ├── mnist-net_256x6.onnx
    ├── mnist/                          # Output directory for MNIST properties
    ├── emnist/                         # Output directory for EMNIST properties
    └── mnist_pixmix_ood/              # Output directory for OOD properties
```

## Prerequisites

### Required Python Packages

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `onnxruntime`
- `PIL` (Pillow, for PixMix script)

### Dataset Requirements

The scripts expect datasets to be located at:
- `../../datasets/MNIST/` - MNIST dataset
- `../../datasets/EMNIST/` - EMNIST dataset

The datasets will be automatically downloaded if not present.

### ONNX Models (Optional)

For verification of correctly classified samples, the scripts can use ONNX models:
- `mnist-net_256x2.onnx`
- `mnist-net_256x4.onnx`
- `mnist-net_256x6.onnx`

If these models are not found, the scripts will proceed without verification.

## Scripts

### 1. `generate_properties.py` - Basic MNIST Property Generator

**Purpose**: Generates properties for a fixed number of MNIST test images with multiple epsilon values.

**Usage**:
```bash
cd mnistfc
python generate_properties.py <seed>
```

**Arguments**:
- `seed` (required): Random seed for reproducibility

**Configuration** (edit in script):
- `num_images = 15`: Number of images to generate properties for
- `epsilons = [0.03, 0.05]`: Epsilon perturbation values
- `data_dir = "../../datasets/MNIST"`: Dataset location

**Output**:
- VNNLIB files: `prop_{i}_{eps:.2f}.vnnlib` (e.g., `prop_0_0.03.vnnlib`)
- CSV file: `mnistfc_instances.csv` with absolute paths to all properties

**Features**:
- Selects images that are correctly classified by all three ONNX models
- Generates properties for each epsilon value

---

### 2. `generate_mnist_properties.py` - Enhanced MNIST Generator

**Purpose**: Generates balanced MNIST properties with exactly N samples per class.

**Usage**:
```bash
cd mnistfc
python generate_mnist_properties.py [seed]
```

**Arguments**:
- `seed` (optional): Random seed for reproducibility

**Configuration** (edit in `main()` function):
```python
samples_per_class = 10      # Samples per class (default: 10, total: 100)
epsilons = [1/255, 2/255, 3/255, 4/255]  # Epsilon values
data_dir = "../../datasets/MNIST"
output_dir = "./mnist"      # Output directory
```

**Output**:
- VNNLIB files: `mnist/prop_{i}_{eps}over255.vnnlib`
- CSV files: `mnist/mnistfc_instances_eps{eps}over255.csv` (one per epsilon)

**Features**:
- Balanced sampling: exactly N samples per class
- Optional ONNX model verification
- Separate CSV file for each epsilon value
- Automatic cleanup of existing .vnnlib files

**Example**:
```bash
python generate_mnist_properties.py 42
```

---

### 3. `generate_emnist_properties.py` - EMNIST Property Generator

**Purpose**: Generates properties for EMNIST dataset with configurable splits.

**Usage**:
```bash
cd mnistfc
python generate_emnist_properties.py [seed]
```

**Arguments**:
- `seed` (optional): Random seed for reproducibility

**Configuration** (edit in `main()` function):
```python
samples_per_class = 10
epsilons = [1/255, 2/255, 3/255, 4/255]
data_dir = "../../datasets/EMNIST"
output_dir = "./emnist"
split = "digits"  # EMNIST split to use
```

**EMNIST Split Options**:
- `"byclass"`: 62 classes (0-9, A-Z, a-z)
- `"bymerge"`: 47 classes (merged similar characters)
- `"balanced"`: 47 classes (balanced class distribution)
- `"letters"`: 26 classes (A-Z only)
- `"digits"`: 10 classes (0-9 only)
- `"mnist"`: 10 classes (MNIST-like)

**Output**:
- VNNLIB files: `emnist/prop_{i}_{eps}over255.vnnlib`
- CSV files: `emnist/emnistfc_instances_eps{eps}over255.csv`

**Features**:
- Supports all EMNIST splits
- Balanced sampling per class
- Automatic image rotation and mirroring correction
- Separate CSV file for each epsilon value

**Example**:
```bash
# Generate properties for EMNIST digits
python generate_emnist_properties.py 42

# To use a different split, edit the 'split' variable in the script
```

---

### 4. `generate_pixmix_mnist_properties.py` - PixMix OOD Property Generator

**Purpose**: Generates properties for Out-of-Distribution (OOD) MNIST images from PixMix dataset.

**Usage**:
```bash
cd mnistfc
python generate_pixmix_mnist_properties.py
```

**Configuration** (edit at top of script):
```python
OOD_ROOT = "../../mnist_pixmix_ood"  # Path to PixMix OOD images
OUTPUT_DIR = "./mnist_pixmix_ood"
EPSILONS = [1/255, 2/255, 3/255, 4/255]
TAU = 0.8
SAMPLES_PER_CLASS = 10
NUM_CLASSES = 10
```

**Requirements**:
- PixMix OOD images organized by class: `{OOD_ROOT}/{class_id}/*.png`
- ONNX models must be present (required, not optional)

**Output**:
- VNNLIB files: `mnist_pixmix_ood/ood_{i}_{eps}over255.vnnlib`
- CSV files: `mnist_pixmix_ood/mnist_ood_instances_eps{eps}over255.csv`

**Features**:
- Selects OOD samples where all ONNX models agree on prediction
- Exactly N samples per class
- Generates OOD robustness properties

**Example**:
```bash
# Make sure OOD_ROOT points to your PixMix OOD directory
python generate_pixmix_mnist_properties.py
```

---

## Output Format

### VNNLIB File Format

Each `.vnnlib` file contains:
1. **Input variable declarations**: `X_0` through `X_783` (for 28×28 images)
2. **Output variable declarations**: `Y_0` through `Y_{num_classes-1}`
3. **Input constraints**: Bounds for each pixel: `X_i ∈ [lower, upper]`
4. **Output constraints**: Robustness property (correct class must have highest logit)

### CSV File Format

Each CSV file contains one property path per line:
```
prop_0_1over255.vnnlib
prop_1_1over255.vnnlib
...
```

These CSV files can be used with verification tools to batch process properties.

---

## Customization

### Changing Epsilon Values

Edit the `epsilons` list in each script:
```python
epsilons = [1/255, 2/255, 4/255, 8/255]  # Custom epsilon values
```

### Changing Number of Samples

For balanced generators:
```python
samples_per_class = 20  # Will generate 20 samples per class
```

### Changing Output Directory

```python
output_dir = "./custom_output"  # Change output location
```

### Disabling ONNX Verification

Set `onnx_models = None` in the script to skip verification:
```python
onnx_models = None  # Skip ONNX model verification
```

---

## Troubleshooting

### Dataset Not Found
- Ensure datasets are located at `../../datasets/MNIST/` or `../../datasets/EMNIST/`
- The scripts will attempt to download datasets automatically if missing

### ONNX Models Not Found
- For `generate_properties.py`, `generate_mnist_properties.py`, and `generate_emnist_properties.py`: Scripts will proceed without verification
- For `generate_pixmix_mnist_properties.py`: ONNX models are required and script will fail if missing

### OOD Root Path Error
- Update `OOD_ROOT` in `generate_pixmix_mnist_properties.py` to point to your PixMix OOD directory
- Ensure directory structure: `{OOD_ROOT}/{class_id}/*.png`

### Memory Issues
- Reduce `samples_per_class` or `num_images` if running out of memory
- Process one epsilon value at a time by modifying the `epsilons` list

---

## Examples

### Generate MNIST properties with 10 samples per class
```bash
cd generate_properties/mnistfc
python generate_mnist_properties.py 42
```

### Generate EMNIST letters properties
```bash
cd generate_properties/mnistfc
# Edit generate_emnist_properties.py: set split = "letters"
python generate_emnist_properties.py 42
```

### Generate basic MNIST properties
```bash
cd generate_properties/mnistfc
python generate_properties.py 12345
```

### Generate PixMix OOD properties
```bash
cd generate_properties/mnistfc
# Edit generate_pixmix_mnist_properties.py: set OOD_ROOT path
python generate_pixmix_mnist_properties.py
```

---

## Notes

- All scripts use relative paths, so they should be run from the `mnistfc/` directory
- Generated properties use the VNNLIB format standard for neural network verification
- CSV files contain relative paths to VNNLIB files from the output directory
- Random seeds ensure reproducibility of sample selection

