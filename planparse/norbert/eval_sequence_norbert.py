import argparse
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import torch



def data_collator(batch_input):

    device = "cuda"

    input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in batch_input]
    attention_mask = [torch.tensor(inst["attention_mask"], dtype=torch.long) for inst in batch_input]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    padded_batch_input = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
        
    return padded_batch_input



class TestDataset(Dataset):
    def __init__(self, docs):

        encodings = tokenizer(docs, truncation=True, padding=False, max_length=512)
        
        self.encodings = []

        for i in range(len(encodings["input_ids"])):

            self.encodings.append(
                {
                    "input_ids" : encodings["input_ids"][i],
                    "attention_mask" : encodings["attention_mask"][i],
                }
            )
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]



def eval_sequence_classifier(model, test_loader, all_labels):

    model.eval()
    model.to("cuda")
        
    all_preds = []
    
    for batch_input in tqdm(test_loader, desc="evaluating model"):

        with torch.no_grad():
            logits = model(batch_input["input_ids"], batch_input["attention_mask"]).logits
            preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())    
    
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Accuracy : {accuracy}")
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    args = parser.parse_args()

        
    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    modelname = args.model

    tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True) 
    
    model = AutoModelForSequenceClassification.from_pretrained(
        modelname, 
        trust_remote_code=True,
    )

    label2id = model.config.label2id
    
    test_path = "./formated_data/huge/augmented_generated_with_edge_cases.jsonl"
    
    with open(test_path, "r") as f:
        test_raw = [json.loads(line) for line in f]
        
    documents = [inst["text"] for inst in test_raw]
    labels = [inst["label"] for inst in test_raw]

    fixed_labels = []
    for label in labels:
        if label == []:
            fixed_labels.append("none")

        elif isinstance(label, list):
            fixed_labels.append(label[0])

    print(f"class counts : {np.unique(fixed_labels, return_counts=True)}")

    labels = [label2id[label] for label in fixed_labels]
        
    test_dataloader = DataLoader(TestDataset(documents), batch_size=8, collate_fn=data_collator)

    eval_sequence_classifier(model, test_dataloader, labels)
    






