from typing import Dict
import argparse 
import shutil
import datetime
import json
import csv
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback
from torch import nn
import numpy as np
import torch

from load_data import load_and_format


class SaveLossCallback(TrainerCallback):
    def __init__(self, log_dir):
        
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, "loss_log.csv")
        
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Loss", "Learning Rate", "Epoch", "Global Step", "eval_accuracy"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        
        if logs is not None and "loss" in logs:
            
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.global_step,       
                    logs.get("loss", "N/A"),  
                    logs.get("learning_rate", "N/A"), 
                    state.epoch,             
                    state.global_step,
                    "N/A"
                ])


def data_collator(batch_input):
    
    device = "cuda"

    input_ids = [torch.tensor(inst["input_ids"], dtype=torch.long) for inst in batch_input]
    attention_mask = [torch.tensor(inst["attention_mask"], dtype=torch.long) for inst in batch_input]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1, padding_side="left")
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0, padding_side="left")

    labels = torch.tensor([inst["labels"] for inst in batch_input], dtype=torch.long) if "labels" in batch_input[0] else None

    new_batch_input = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }
    if labels is not None:
        new_batch_input["labels"] = labels.to(device)

    return new_batch_input


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
        model : AutoModelForSequenceClassification,
        tokenizer : AutoTokenizer,
        traindata : CustomDataset,
        evaldata : CustomDataset,
        testdata : CustomDataset,
        trainer_config : Dict,
        ):

    savedir = "./local_models_storage/"
    logdir = os.path.join(trainer_config["logging_dir"], trainer_config["run_name"] + "_" + datetime.datetime.now().strftime('%Y%m%d_%H:%M'))

    print(f"logs and results will be saved to : {logdir}")
    
    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")
    
    trainerargs = TrainingArguments(
        per_device_train_batch_size=trainer_config["per_device_train_batch_size"],
        gradient_accumulation_steps=trainer_config["gradient_accumulation_steps"],
        per_device_eval_batch_size=trainer_config["per_device_eval_batch_size"],
        torch_empty_cache_steps=trainer_config["torch_empty_cache_steps"],
        lr_scheduler_type=trainer_config["lr_scheduler_type"],
        num_train_epochs=trainer_config["num_train_epochs"],
        eval_strategy=trainer_config["eval_strategy"],
        learning_rate=trainer_config["learning_rate"],
        logging_steps=trainer_config["logging_steps"],
        save_strategy=trainer_config["save_strategy"],
        warmup_steps=trainer_config["warmup_steps"],
        weight_decay=trainer_config["weight_decay"],
        logging_dir=trainer_config["logging_dir"],
        output_dir=trainer_config["output_dir"],
        eval_steps=trainer_config["eval_steps"],
        max_steps=trainer_config["max_steps"],
        report_to=trainer_config["report_to"],
        data_seed=trainer_config["data_seed"],
        run_name=trainer_config["run_name"],
        optim=trainer_config["optim"],
        seed=trainer_config["seed"],
    )

    trainerargs = trainerargs.set_dataloader(pin_memory=False)
    
    trainerargs.per_device_train_batch_size = trainer_config["per_device_train_batch_size"]
    trainerargs.per_device_eval_batch_size = trainer_config["per_device_eval_batch_size"]

    def compute_metrics(eval_preds):
    
        logits, true_labels = eval_preds
        predictions = np.argmax(logits, axis=1)
        
        accuracy = accuracy_score(true_labels, predictions)

        with open(os.path.join(logdir, "loss_log.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                accuracy
            ])

        return {"accuracy": accuracy}
    
    trainer = Trainer(
        callbacks=[SaveLossCallback(logdir)],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=traindata,
        eval_dataset=evaldata,
        tokenizer=tokenizer,
        args=trainerargs,
        model=model,
    )

    metrics = trainer.train()[2]

    print("Evaluating on test set")
    res = trainer.evaluate(testdata)

    metrics["test_accuracy"] = res["eval_accuracy"]

    with open(os.path.join(logdir, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")

    save_path = os.path.join(savedir, "norbert-seqcls-{}".format(datetime.datetime.now().strftime("%Y%m%d_%H:%M")))



if __name__ == "__main__":

    if not os.path.isfile("./setup.py"):
        raise Exception("Please run from the root of the repository")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action='store_true', help="Run a small dataset for testing")
    parser.add_argument("--config", type=str, help="path to model config", required=True)
    args = parser.parse_args()

    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    with open(args.config, "r") as f:
        config = json.load(f)

    model_config = config["model_config"]
    trainer_config = config["trainer_config"]

    model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_config["huggingface_model"],
            id2label={v:k for k,v in model_config["label2id"].items()}, 
            trust_remote_code=model_config["trust_remote_code"],
            num_labels=len(model_config["label2id"]), 
            label2id=model_config["label2id"],
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_config["huggingface_model"],
        )

    if args.smoke:
        train_path = "./formated_data/smoke/train_smoke.jsonl"
        eval_path = "./formated_data/smoke/eval_smoke.jsonl"
        test_path = "./formated_data/smoke/eval_smoke.jsonl"
    
    else:
        train_path = config["train_path"]
        eval_path = config["eval_path"]
        test_path = config["test_path"]
    
    train_x, train_y = load_and_format(train_path, label2id=model_config["label2id"])
    eval_x, eval_y = load_and_format(eval_path, label2id=model_config["label2id"])
    test_x, test_y = load_and_format(test_path, label2id=model_config["label2id"])

    tokenized_train_x = tokenizer(train_x, truncation=True, padding=False, max_length=512)
    tokenized_eval_x = tokenizer(eval_x, truncation=True, padding=False, max_length=512)
    tokenized_test_x = tokenizer(test_x, truncation=True, padding=False, max_length=512)

    train_dataset = CustomDataset(tokenized_train_x, train_y)
    eval_dataset = CustomDataset(tokenized_eval_x, eval_y)
    test_dataset = CustomDataset(tokenized_test_x, test_y)

    train(
        trainer_config=trainer_config,
        traindata=train_dataset,
        evaldata=eval_dataset,
        testdata=test_dataset,
        tokenizer=tokenizer,
        model=model,
    )


