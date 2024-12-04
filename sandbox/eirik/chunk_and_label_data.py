import os
import re
import json

from transformers import AutoTokenizer


def is_integer(string):

    try:
        int(string)  
        return True
    except ValueError:
        return False

def print_doc(long_string):
        words = long_string.split()  
        for i in range(0, len(words), 10):
            print(" ".join(words[i:i + 10]))
        print()


def chunk_text():

    datadir = "./data"
    
    data_save_path = "./formated_data/norbert_sequence_labeled_chunks.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-large")

    multiple_whitespaces = re.compile(r'\s\s+')
    
    for f in os.listdir(datadir):

        document_path = os.path.join(datadir, f, f"{f}.txt")
        label_path = os.path.join(datadir, f, "y.json")

        with open(label_path, "r") as f:
            labels = json.load(f)["utnyttingsgrad"]
            print()

        with open(document_path, "r") as f:

            text = f.read()
            text = multiple_whitespaces.sub(' ', text)
            
            tokenized_text = tokenizer.tokenize(text)

            n_chunks = len(tokenized_text) // 256
            
            text_len = len(text.split())

            step = text_len // n_chunks
            start = 0
            end = step 
            for i in range(n_chunks):
                
                if end > text_len:
                    end = text_len

                text_chunk = " ".join(text.split()[start:end])
                print()
                print(labels)
                print()
                print_doc(text_chunk)
                
                start = end
                end = start + step

                is_class = input("is utnyttingsgrad in this chunk? all/index/none: ")

                if is_class == "all":

                    data = {
                        "text": text_chunk,
                        "label": labels
                    }

                elif is_integer(is_class):

                    data = {
                        "text": text_chunk,
                        "label": labels[int(is_class)]
                    }

                else:

                    data = {
                        "text": text_chunk,
                        "label": []
                    }

                with open(data_save_path, "a") as f:
                    f.write(json.dumps(data) + "\n")
                            
                    


if __name__ == "__main__":
    chunk_text()