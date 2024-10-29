#!/bin/bash

# assure that the file is run from root of the repository
if [ "$(basename "$PWD")" != "planparse" ]; then
    echo "Error: You are not in the project root directory (planparse)."
    echo "Please navigate to the correct directory and run this script again."
    exit 1
fi

mkdir -p access_token
touch access_token/token.txt

# Update package list
sudo apt-get update

# Install Tesseract OCR and dependencies
sudo apt-get install -y \
    libleptonica-dev \
    tesseract-ocr \
    libtesseract-dev \
    python3-pil \
    tesseract-ocr-eng \
    tesseract-ocr-script-latn \
    tesseract-ocr-nor