#!/usr/bin/env bash
# build.sh
set -e

echo "Installing system dependencies..."
apt-get update -y
apt-get install -y tesseract-ocr libgl1-mesa-glx libglib2.0-0

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Build complete!"