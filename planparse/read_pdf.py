import os
import pdf2image
import pytesseract

from PyPDF2 import PdfReader
from pytesseract import Output, TesseractError

def read_pdf(pdf_path):
    try:
        images = pdf2image.convert_from_path(pdf_path)
        full_text = ""
        for image in images:
            ocr_dict = pytesseract.image_to_data(image, lang='nor', output_type=Output.DICT)
            text = " ".join(ocr_dict['text'])
            full_text += "\n" + text + "\n"
        return full_text.strip()
    except TesseractError as e:
        print(f"OCR failed: {e}")
        return ""
    except Exception as e:
        print(f"Error processing image PDF: {e}")
        return ""

'''
def read_txt_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        return full_text.strip()
    except Exception as e:
        print(f"Error reading text from PDF: {e}")
        return ""

def read_mixed_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pdf_text = ""
    
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        
        if page_text:
            pdf_text += page_text + "\n"
        else:
            try:
                # Convert this specific page to an image and apply OCR
                images = pdf2image.convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                for image in images:
                    ocr_text = pytesseract.image_to_string(image, lang='nor')
                    pdf_text += ocr_text + "\n"
            except Exception as e:
                print(f"Error processing page {i + 1}: {e}")
                
    return pdf_text.strip()'''
