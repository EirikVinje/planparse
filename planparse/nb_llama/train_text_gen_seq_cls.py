"""
Implement a class for sequence classification. E.g to ltg/norbert3-large (https://huggingface.co/ltg/norbert3-large)
"""
from pprint import pprint
from typing import Dict
import logging
import argparse 
import shutil
import datetime
import json
import csv
import os

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import SequentialSampler, Dataset, DataLoader
from transformers.trainer_callback import TrainerCallback
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import torch

from llm_base_text_gen_seq_cls import CausalTextClSModel
from planparse.prompter import Prompter
from load_data import load_and_format

logger = logging.getLogger("planparse")
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


class SaveLossCallback(TrainerCallback):
    
    def __init__(self, log_dir):
        
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, "loss_log.csv")
        
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Loss", "Learning Rate", "Epoch", "Global Step", "accuracy", "mean_precision"])

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
                    "N/A",
                    "N/A",
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
        model : CausalTextClSModel,
        traindata : CustomDataset,
        evaldata : CustomDataset,
        testdata : CustomDataset,
        testdata2 : CustomDataset,
        config : Dict
        ):

    config = config["trainer_config"]

    savedir = "./local_models_storage/"
    logdir = os.path.join(config["logging_dir"], config["run_name"] + "_" + datetime.datetime.now().strftime('%Y%m%d_%H:%M'))

    print(f"logs and results will be saved to : {logdir}")
    
    if os.path.isdir("./temp"):
        shutil.rmtree("./temp")
    
    trainerargs = TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        torch_empty_cache_steps=config["torch_empty_cache_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        num_train_epochs=config["num_train_epochs"],
        eval_strategy=config["eval_strategy"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_dir=config["logging_dir"],
        output_dir=config["output_dir"],
        eval_steps=config["eval_steps"],
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

    def compute_metrics(eval_preds):
    
        logits, true_labels = eval_preds
        predictions = np.argmax(logits, axis=1)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="macro")

        with open(os.path.join(logdir, "loss_log.csv"), "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                accuracy,
                precision
            ])

        return {"accuracy": accuracy, "mean_precision" : precision}
    
    trainer = Trainer(
        callbacks=[SaveLossCallback(logdir)],
        compute_metrics=compute_metrics,
        processing_class=model.tokenizer,
        data_collator=data_collator,
        train_dataset=traindata,
        eval_dataset=evaldata,
        args=trainerargs,
        model=model,
    )

    metrics = trainer.train()[2]

    print("Evaluating on test set")
    res = trainer.evaluate(testdata)
    metrics["test_accuracy"] = res["eval_accuracy"]
    metrics["test_mean_precision"] = res["eval_mean_precision"]

    res2 = trainer.evaluate(testdata2)
    metrics["test_accuracy2"] = res2["eval_accuracy"]
    metrics["test_mean_precision2"] = res2["eval_mean_precision"]

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

    if args.smoke:
        train_path = "./formated_data/smoke/train_smoke.jsonl"
        eval_path = "./formated_data/smoke/eval_smoke.jsonl"
        test_path = "./formated_data/smoke/eval_smoke.jsonl"
    
    else:
        train_path = config["train_path"]
        eval_path = config["eval_path"]
        test_path = config["test_path"]
        test_path2 = "./formated_data/huge/augmented_generated_with_edge_cases.jsonl"

    label2id = config["label2id"]
    
    prompter = Prompter(
        template_path = config["template_path"],   
    )
    prompter.load()

    model = CausalTextClSModel(config=config, n_labels=len(label2id))
    
    train_x, train_y = load_and_format(train_path, prompt_generator=prompter, label2id=label2id)
    eval_x, eval_y = load_and_format(eval_path, prompt_generator=prompter, label2id=label2id)
    test_x, test_y = load_and_format(test_path, prompt_generator=prompter, label2id=label2id)
    test_x2, test_y2 = load_and_format(test_path2, prompt_generator=prompter, label2id=label2id)

    # print("train labels: ", np.unique(train_y, return_counts=True))
    # print("eval labels: ", np.unique(eval_y, return_counts=True))
    # print("test labels: ", np.unique(test_y, return_counts=True))

    tokenized_train_x = model.tokenizer(train_x, truncation=True, padding=False)
    tokenized_eval_x = model.tokenizer(eval_x, truncation=True, padding=False)
    tokenized_test_x = model.tokenizer(test_x, truncation=True, padding=False)
    tokenized_test_x2 = model.tokenizer(test_x2, truncation=True, padding=False)

    train_dataset = CustomDataset(tokenized_train_x, train_y)
    eval_dataset = CustomDataset(tokenized_eval_x, eval_y)
    test_dataset = CustomDataset(tokenized_test_x, test_y)
    test_dataset2 = CustomDataset(tokenized_test_x2, test_y2)
    
    train(
            config=config,
            traindata=train_dataset,
            evaldata=eval_dataset,
            testdata=test_dataset,
            testdata2=test_dataset2,
            model=model,
        )





