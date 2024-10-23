import json
import os
import re

from pdf2image import convert_from_path
from tqdm import tqdm
from PIL import Image
import pytesseract


def parse_text(text):

    remove_newlines = re.compile(r'\n+')
    remove_multiple_whitespaces = re.compile(r'\s\s+')

    text = remove_newlines.sub(' ', text)
    text = remove_multiple_whitespaces.sub(' ', text)
    text = text.strip()

    return text




def extract_text_from_pdf(pdf_path : str, extract_to_path : str, output_file : str):
    """
    Extract text from a PDF containing scanned images using OCR.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from all pages
    """
    # Convert PDF pages to images
    images = convert_from_path(pdf_path)
    
    # Create a temporary directory for image files if needed
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)
    
    extracted_text = []

    # Process each page
    for i, image in tqdm(enumerate(images), desc="Extracting text from pages"):
        # Save page as temporary image
        temp_path = os.path.join(extract_to_path, f'page_{i}.png')
        image.save(temp_path, 'PNG')
        
        # Perform OCR on the image
        text = pytesseract.image_to_string(Image.open(temp_path), lang="nor", config="--oem 3 --psm 3")
        extracted_text.append(text)
        
        os.remove(temp_path)
        
    content = parse_text(" ".join(extracted_text))

    save_dict = {"original_file" : pdf_path, "extracted_text" : content}

    output_path = os.path.join(extract_to_path, output_file)

    with open(output_path, 'w') as f:
        json.dump(save_dict, f, indent=4)


# Example usage
if __name__ == "__main__":
    
    text = extract_text_from_pdf("./ex_data/200921.pdf", "./imagedata/", "200921.json")