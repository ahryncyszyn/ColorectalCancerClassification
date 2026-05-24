#!/bin/bash
set -e

CONFIG_DIR="configs"

for config_file in "$CONFIG_DIR"/*.yaml; do
    echo "========================================"
    echo "Starting: $config_file"
    echo "========================================"
    PYTHONPATH=. python scripts/train.py --config "$config_file"
    echo ""
done