#!/bin/bash

# Update package list
sudo apt-get update

# Install Tesseract OCR and dependencies
sudo apt-get install -y \
    libleptonica-dev \
    tesseract-ocr \
    tesseract-ocr-dev \
    libtesseract-dev \
    python3-pil \
    tesseract-ocr-eng \
    tesseract-ocr-script-latn \
    tesseract-ocr-nor
