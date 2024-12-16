import os
import re
import shutil
import argparse
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
        
        # Remove special signs except "=" and "%"
        full_text = re.sub(r"[^\w\s=%.]", " ", full_text)

        # Remove single characters that are not "=" or "%"
        full_text = re.sub(r"\b(?![=%])\w\b", "", full_text)
        #full_text = re.sub(r"(?<!\w)\.(?!\w)", "", full_text)
        full_text = full_text.replace("m*", "m2") 
        full_text = full_text.replace("m?", "m2") 

        # Remove extra white spaces
        full_text = re.sub(r"\s+", " ", full_text).strip()

        return full_text
    
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

                shutil.copyfile(pdf_path, f"{save_folder}/{base_name}/{base_name}.pdf")
    print("\n")
    print("Finished processing PDFs")
    print(f"Text files saved in {save_folder}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="Path to PDF file", default="ex_data")
    parser.add_argument("--save_folder", type=str, help="Folder to save text files", default="data")
    args = parser.parse_args()
    pdf_path = args.pdf_path
    save_folder = args.save_folder

    pdf_files = glob.glob(f"{pdf_path}/*.pdf")
    print("These are the pdf files that will be parsed: \n", pdf_files)
    read_multiple_pdfs(pdf_files, save_folder)