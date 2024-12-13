"""
Implement a class for sequence classification. E.g to ltg/norbert3-large (https://huggingface.co/ltg/norbert3-large)
"""
from typing import Dict
import argparse 
import shutil
import datetime
import json
import os

from pprint import pprint

from torch.utils.data import SequentialSampler, Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig
from peft import get_peft_model
from torch import nn
import numpy as np
import evaluate
import torch

from llm_base_text_gen_seq_cls import CausalTextClSModel
from planparse.prompter import Prompter


class Trainer(Trainer):
    def _get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,  # Ensure sequential
            collate_fn=self.data_collator
        )

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


def compute_metrics(eval_preds):
    
    metric = evaluate.load("accuracy")
    logits, true_labels = eval_preds

    predictions = np.argmax(logits, axis=1)
    
    output = {
        "accuracy": metric.compute(predictions=predictions, references=true_labels)['accuracy'],
    }
    
    return output


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



def train(
        model : CausalTextClSModel,
        traindata : CustomDataset,
        evaldata : CustomDataset,
        config : Dict
        ):

    config = config["trainer_config"]

    savedir = "./local_models_storage/"

    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")
    
    trainerargs = TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        torch_empty_cache_steps=config["torch_empty_cache_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        num_train_epochs=config["num_train_epochs"],
        eval_strategy=config["eval_strategy"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        output_dir=config["output_dir"],
        max_steps=config["max_steps"],
        report_to=config["report_to"],
        data_seed=config["data_seed"],
        run_name=config["run_name"],
        optim=config["optim"],
        seed=config["seed"],
    )

    trainerargs = trainerargs.set_dataloader(pin_memory=False)
    
    trainerargs.per_device_train_batch_size = config["per_device_train_batch_size"]
    trainerargs.per_device_eval_batch_size = config["per_device_eval_batch_size"]

    trainer = Trainer(
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
        train_dataset=traindata,
        eval_dataset=evaldata,
        args=trainerargs,
        model=model,
    )

    metrics = trainer.train()
    print(metrics)
    

    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")

    # save_path = os.path.join(savedir, "norbert-seqcls-{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    # print(f"Model and tokenizer saved to : {save_path}")    




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
        train_path = config["train_path"]
        eval_path = config["eval_path"]

    label2id = config["label2id"]
    
    prompter = Prompter(
        template_path = config["template_path"],   
    )
    prompter.load()

    with open(train_path, "r") as f:
        train_raw = [json.loads(line) for line in f]

    train_x = [inst["text"] for inst in train_raw]
    train_y = [inst["label"] for inst in train_raw]
    train_x = [prompter(inst) for inst in train_raw]
    
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
    eval_x = [prompter(inst) for inst in eval_x]

    fixed_labels = []
    for label in eval_y:
        if label == []:
            fixed_labels.append("none")
        elif isinstance(label, list):
            fixed_labels.append(label[0])
    eval_y = [label2id[label] for label in fixed_labels] 

    model = CausalTextClSModel(config=config, n_labels=len(label2id))

    model.to_device()

    tokenized_train_x = model.tokenizer(train_x, truncation=True, padding=False)
    train_dataset = CustomDataset(tokenized_train_x, train_y)
    
    tokenized_eval_x = model.tokenizer(eval_x, truncation=True, padding=False)
    eval_dataset = CustomDataset(tokenized_eval_x, eval_y)

    train(model, train_dataset, eval_dataset, config)
    





