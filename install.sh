#!/usr/bin/env bash
set -e  # exit on first error

echo "=== Setup Conda environment (traffic) ==="

# Create env if not exists, otherwise update
if conda env list | grep -qE "^\s*traffic\s"; then
    echo "Environment 'traffic' already exists. Updating..."
    conda env update -f traffic.yml --name traffic --prune
else
    echo "Creating new environment 'traffic'..."
    conda env create -f traffic.yml --name traffic
fi

echo "=== Installing SUMO ==="
# Add SUMO stable PPA only if not already present
if ! grep -q "^deb .\+sumo/stable" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
    sudo add-apt-repository -y ppa:sumo/stable
fi

sudo apt-get update -qq
sudo apt-get install -y sumo sumo-tools sumo-doc

echo "=== Done! ==="
