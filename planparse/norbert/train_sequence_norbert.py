"""
Implement a class for filling masks. E.g to ltg/norbert3-large (https://huggingface.co/ltg/norbert3-large)
"""

import argparse 
import shutil
import datetime
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch


def custom_data_collator(batch_input):
    
    device = "cuda"

    input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in batch_input]
    attention_mask = [torch.tensor(inst["attention_mask"], dtype=torch.long) for inst in batch_input]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    labels = torch.tensor([inst["labels"] for inst in batch_input], dtype=torch.long) if "labels" in batch_input[0] else None

    new_batch_input = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    if labels is not None:
        new_batch_input["labels"] = labels.to(device)

    return new_batch_input


class TrainDataset(Dataset):
    def __init__(self, encodings, labels):

        self.encodings = []

        for i in range(len(encodings["input_ids"])):

            self.encodings.append(
                {
                    "input_ids" : encodings["input_ids"][i],
                    "attention_mask" : encodings["attention_mask"][i],
                    "labels" : labels[i],
                }
            )
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]
    

def train(
        model : AutoModelForSequenceClassification,
        tokenizer : AutoTokenizer,
        traindata : TrainDataset, 
        ):

    savedir = "./local_models_storage/"
    
    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")
    
    trainerargs = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        torch_empty_cache_steps=True,
        lr_scheduler_type="linear",
        num_train_epochs=10,
        run_name="run_norbert",
        learning_rate=0.001,
        logging_steps=10000,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir="./temp",
        max_steps=-1,
        data_seed=42,
        optim="adamw_torch",
        seed=42,
        report_to="none",
    )

    training_args = trainerargs.set_dataloader(pin_memory=False)

    model.train()
    
    trainer = Trainer(
        data_collator=custom_data_collator,
        train_dataset=traindata,
        args=trainerargs,
        model=model,
    )

    
    
    trainer.train()
    
    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")

    save_path = os.path.join(savedir, "norbert-seqcls-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model and tokenizer saved to : {save_path}")    


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    classes=[
        "BYA-87", #0
        "BRA-69", #1
        "TU", #2
        "U", #3
        "F", #4
        "BGA", #5
        "BFA", #6
        "%-BYA-97", #7 
        "T-BRA", #8
        "%-TU", #9
        "%-BYA", #10
        "BYA", #11
        "BRA", #12
        "%-BRA", #13
    ]

    include_idx = [10, 11, 12, 13]
    include_classes = [classes[i] for i in include_idx]

    id2label = {i+1: c for i, c in enumerate(include_classes)}
    label2id = {c: i+1 for i, c in enumerate(include_classes)}

    id2label[0] = "none"
    label2id["none"] = 0
    
    model = AutoModelForSequenceClassification.from_pretrained(
            "ltg/norbert3-large", 
            num_labels=len(label2id.keys()), 
            id2label=id2label, 
            label2id=label2id,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-large")

    train_path = "./formated_data/train_generated_dataset.jsonl"
    
    with open(train_path, "r") as f:
        train_raw = [json.loads(line) for line in f]

    documents = [inst["text"] for inst in train_raw]
    labels = [inst["label"] for inst in train_raw]

    fixed_labels = []
    for label in labels:
        if label == []:
            fixed_labels.append("none")

        elif isinstance(label, list):
            fixed_labels.append(label[0])

    labels = [label2id[label] for label in fixed_labels]
    
    tokenized_docs = tokenizer(documents, truncation=True, padding=False, max_length=512)

    train_dataset = TrainDataset(tokenized_docs, labels)

    train(model, tokenizer, train_dataset)











