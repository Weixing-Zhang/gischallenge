#!/bin/bash

# Author: Weixing Zhang
# Date: 07/20/2021 
# Description: This script is used to setup a conda environment

# Exit when error occurs
set -e

# Create conda environment
export ENV_NAME=$@
conda create -n $ENV_NAME -y --channel conda-forge

# Activate the environment in Bash shell
conda activate $ENV_NAME

# Conda install dependencies
conda install --file conda-requirements.txt -y --channel conda-forge

# Pip install the rest dependencies
pip install -r pip-requirements.txt

echo "===== Dependencies installment completes. ====="
