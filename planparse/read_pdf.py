import os
import shutil
import pdf2image
import pytesseract

from PyPDF2 import PdfReader
from pytesseract import Output, TesseractError

import glob

def read_pdf(pdf_path):
    try:
        images = pdf2image.convert_from_path(pdf_path)
        full_text = ""
        for image in images:
            ocr_dict = pytesseract.image_to_data(image, lang='nor', output_type=Output.DICT)
            text = " ".join(ocr_dict['text'])
            full_text += " " + text
        return full_text.strip()
    except TesseractError as e:
        print(f"OCR failed: {e}")
        return ""
    except Exception as e:
        print(f"Error processing image PDF: {e}")
        return ""


def read_multiple_pdfs(pdf_paths, save_folder):
    save_folder = save_folder
    for pdf_path in pdf_paths:
        if os.path.isfile(pdf_path):
            text = read_pdf(pdf_path)
            if text:
                base_name = os.path.basename(pdf_path).replace(".pdf", "")
                new_folder = f"{save_folder}/{base_name}"
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                save_path = os.path.join(new_folder, base_name + ".txt")
                # delete file if it already exists
                if os.path.isfile(save_path):
                    os.remove(save_path)
                with open(save_path, 'a') as f:
                    f.write(text)
                    f.write("\n")

                shutil.copyfile(pdf_path, f"{save_folder}/{base_name}/{base_name}.pdf")
                
# list all pdf files in /ex_data/
pdf_files = glob.glob("ex_data/*.pdf")
print(pdf_files)
read_multiple_pdfs(pdf_files, "data/")