#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH_DIR="$SCRIPT_DIR/../alpha_beta_CROWN/complete_verifier/sh"

cd "$SH_DIR"

# Generate YAML configs for each dataset
bash generate_yml_configs_mnist.sh
bash generate_yml_configs_emnist.sh
bash generate_yml_configs_pixmix.sh

# Run verification for each dataset
bash run_all_configs.sh --pattern "mnist_fc" --continue-on-error --config-dir exp_configs/generalizability/mnist        --log-dir ../expr03_logs/mnist
bash run_all_configs.sh --pattern "mnist_fc" --continue-on-error --config-dir exp_configs/generalizability/emnist       --log-dir ../expr03_logs/emnist
bash run_all_configs.sh --pattern "mnist_fc" --continue-on-error --config-dir exp_configs/generalizability/mnist_pixmix_ood --log-dir ../expr03_logs/pixmix
