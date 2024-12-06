import json
import os
import re

from tqdm import tqdm
import spacy


def generate(root_dir : str, save_dir : str):
    
    nlp = spacy.blank("nb")
    nlp.add_pipe("sentencizer")

    multiple_whitespaces = re.compile(r'\s\s+')

    save_path = os.path.join(save_dir, "phrased_dataset_mlm.jsonl")    

    for file in tqdm(os.listdir(root_dir), desc="Phrasing documents"):

        file_path = os.path.join(root_dir, file, f"{file}.txt")

        with open(file_path, "r") as f:
            text = f.read()

        text = multiple_whitespaces.sub(' ', text)

        doc = nlp(text)

        for sent in tqdm(doc.sents):
            
            data = {
                "phrase" : sent.text,
            }

            with open(save_path, "a") as f:
                f.write(json.dumps(data) + "\n")
            

if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")

    root_dir = "./data"
    save_dir = "./formated_data"

    generate(root_dir, save_dir)