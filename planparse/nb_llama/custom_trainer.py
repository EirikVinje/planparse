import argparse 
import shutil
import datetime
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch import nn
import numpy as np
import evaluate
import torch

from llm_base_text_gen_seq_cls import CausalTextClSModel
from planparse.prompter import Prompter


class CustomDataset(Dataset):
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
        
        self.encodings = sorted(self.encodings, key=lambda x: len(x["input_ids"]), reverse=False)
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def data_collator(batch_input):
    
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


def train_model(
    model: CausalTextClSModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader
):
    """
    Custom training loop for the CausalTextClSModel.

    Args:
        model (CausalTextClSModel): Model to train.
        train_dataloader (DataLoader): Dataloader for the training data.
        val_dataloader (DataLoader): Dataloader for the validation data.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_epochs = 5
    max_grad_norm = 0.8
    log_interval = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    global_step = 0
    acc = 0.0
    
    with tqdm(total=num_epochs * len(train_dataloader), desc=f"(epoch : None) - (loss : None) - (accuracy : None) - (lr : None)") as pbar:

        for epoch in range(num_epochs):
            
            for step, batch in enumerate(train_dataloader):
                
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                loss.backward()
                
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()

                pbar.set_description(f"(epoch : {epoch}) - (loss : {loss.item():.4f}) - (accuracy : {acc:.4f}) - (lr : {optimizer.param_groups[0]['lr']:.4f})")

                global_step += 1
                pbar.update(1)

            # acc = evaluate_model(model, val_dataloader, device)

            scheduler.step()
            
            pbar.set_description(f"(epoch : {epoch}) - (loss : {loss.item():.4f}) - (accuracy : {acc:.4f}) - (lr : {optimizer.param_groups[0]['lr']:.4f})")


def evaluate_model(
        model: CausalTextClSModel, 
        dataloader: DataLoader, 
        device: str = "cuda"
        ):
    
    """
    Evaluate the model on a validation dataset.

    Args:
        model (CausalTextClSModel): Model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to run evaluation on.

    """

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    
    return accuracy





if __name__ == "__main__":

    if not os.path.isfile("./setup.py"):
        raise Exception("Please run from the root of the repository")

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action='store_true', help="Run a small dataset for testing")
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    with open("./configs/llama_1b.json", "r") as f:
        config = json.load(f)

    if args.smoke:
        train_path = "./formated_data/smoke/train_smoke.jsonl"
        eval_path = "./formated_data/smoke/eval_smoke.jsonl"
    
    else:
        train_path = "./formated_data/huge/splits/train.jsonl"
        eval_path = "./formated_data/huge/splits/eval.jsonl"

    label2id = {
        "none" : 0,
        "%-BYA" : 1,
        "BYA" : 2,
        "BRA" : 3,
        "%-BRA" : 4,
    }

    with open(train_path, "r") as f:
        train_raw = [json.loads(line) for line in f]

    train_x = [inst["text"] for inst in train_raw]
    train_y = [inst["label"] for inst in train_raw]

    prompter = Prompter(
        template_path = config["template_path"],   
    )
    
    prompter.load()

    train_x = [prompter(inst, None) for inst in train_x]

    fixed_labels = []
    for label in train_y:
        if label == []:
            fixed_labels.append("none")
        elif isinstance(label, list):
            fixed_labels.append(label[0])

    train_y = [label2id[label] for label in fixed_labels]

    with open(eval_path, "r") as f:
        eval_raw = [json.loads(line) for line in f]

    eval_x = [inst["text"] for inst in eval_raw]
    eval_y = [inst["label"] for inst in eval_raw]

    prompter = Prompter(
        template_path = config["template_path"],   
    )
    
    prompter.load()

    train_x = [prompter(inst, None) for inst in eval_x]

    fixed_labels = []
    for label in eval_y:
        if label == []:
            fixed_labels.append("none")
        elif isinstance(label, list):
            fixed_labels.append(label[0])

    eval_y = [label2id[label] for label in fixed_labels]

    model = CausalTextClSModel(config=config, n_labels=len(label2id))

    model.load_base_model()
    model.init_seq_cls_head()

    model.to("cuda")
    model.train()

    tokenized_train_x = model.tokenizer(train_x, truncation=True, padding=False)
    train_dataset = CustomDataset(tokenized_train_x, train_y)
    
    tokenized_eval_x = model.tokenizer(eval_x, truncation=True, padding=False)
    eval_dataset = CustomDataset(tokenized_eval_x, eval_y)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)    

    train_model(model, train_dataloader, eval_dataloader)