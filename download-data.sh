#!/bin/bash

# Author: Weixing Zhang
# Date: 07/20/2021 
# Description: This script is used to download the provided LAS files

# Exit when error occurs
set -e

# Create a folder to store the data if the folder does not exist yet
mkdir -p download 

# Download the LAS files
echo "===== Downloading C_37EZ1 (~2GB) ====="
wget https://download.pdok.nl/rws/ahn3/v1_0/laz/C_37EZ1.LAZ -O ./download/C_37EZ1.LAZ 

echo "===== Downloading C_37EZ2 (~2.16GB) ====="
wget https://download.pdok.nl/rws/ahn3/v1_0/laz/C_37EZ2.LAZ -O ./download/C_37EZ2.LAZ

echo "===== Downloading completes. ====="
