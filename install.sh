#!/usr/bin/env bash

# Linux/macOS installer script for LayTracer
# This script creates a Conda environment based on environment.yml and installs the package
# Run: ./install.sh
# D. Anikiev, 2026-03-03

set -u

ENV_NAME="laytracer"
PACKAGE_NAME="laytracer"
ENV_YAML="environment.yml"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not installed or not found in PATH."
    echo "Please install Conda (or Miniforge) and try again."
    exit 2
fi

if [ ! -f "$ENV_YAML" ]; then
    echo "$ENV_YAML not found in the current directory."
    echo "Please run this script from the repository root."
    exit 3
fi

echo "Creating Conda environment $ENV_NAME from $ENV_YAML..."
conda env create -f "$ENV_YAML"
if [ $? -ne 0 ]; then
    echo "Failed to create Conda environment $ENV_NAME."
    exit 4
fi

echo "Conda environments:"
conda env list

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Could not locate conda.sh for environment activation."
    exit 5
fi

echo "Activating Conda environment $ENV_NAME..."
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment $ENV_NAME."
    exit 5
fi

echo "Installing $PACKAGE_NAME..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "Failed to install $PACKAGE_NAME."
    exit 6
fi

echo "Python version:"
python --version

echo "Python path:"
command -v python

echo "Done!"
