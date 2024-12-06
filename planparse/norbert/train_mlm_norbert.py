"""
Implement a class for filling masks. E.g to ltg/norbert3-large (https://huggingface.co/ltg/norbert3-large)
"""

import argparse 
import shutil
import datetime
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer
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
        model : AutoModelForMaskedLM,
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

    save_path = os.path.join(savedir, "norbert-domain-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"Model and tokenizer saved to : {save_path}")    


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    
    model = AutoModelForMaskedLM.from_pretrained(
            "ltg/norbert3-large", 
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-large")

    train_path = "./formated_data/train_generated_dataset.jsonl"
    
    with open(train_path, "r") as f:
        train_raw = [json.loads(line) for line in f]
    
    # tokenized_docs = tokenizer(documents, truncation=True, padding=False, max_length=512)

    # train_dataset = TrainDataset(tokenized_docs, labels)

    # train(model, tokenizer, train_dataset)











