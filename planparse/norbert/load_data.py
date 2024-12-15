import json


from planparse.prompter import Prompter



def load_data(
        data_path : str,
        filtype : str = "jsonl",
        ):
    
    if filtype == "jsonl":

        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                yield data["text"], data["label"]

    else:
        raise ValueError(f"Invalid file type {filtype}")


def load_and_format(
        data_path : str,
        filtype : str = "jsonl",
        label2id : dict = None,
        prompt_generator : Prompter = None,
        ):
    
    formated_texts = []
    labels = []

    for text, label in load_data(data_path, filtype):
        
        if prompt_generator is None:
            formated_texts.append(text)
    
        else:
            formated_texts.append(prompt_generator(text))
        
        labels.append(label)

    if label2id is not None:
        labels = [label2id[label[0]] if label != [] else 0 for label in labels]

    return formated_texts, labels