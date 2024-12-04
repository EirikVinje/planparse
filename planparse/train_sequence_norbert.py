"""
Implement a class for filling masks. E.g to ltg/norbert3-large (https://huggingface.co/ltg/norbert3-large)
"""
import argparse 
import shutil
import datetime
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch


from generate_fill_mask_config import generate_config




class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}



class SequenceClassifier:
    def __init__(
            self,
            modelname, 
            num_classes,
            label2id,
            id2label,
            ):

        self.model = AutoModelForSequenceClassification.from_pretrained(
            modelname, 
            num_labels=2, 
            id2label=id2label, 
            label2id=label2id
        )
    
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)


def load_data(
        file_path : str = "./formated_data/norbert_sequence_labeled_chunks.jsonl",        
        ):
    
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]
    



def train(
        llm : any, 
        config : dict, 
        traindata : list[dict], 
        ):

    trainer_config = config["trainer_config"]

    if os.path.isdir(trainer_config["output_dir"]):
        shutil.rmtree(trainer_config["output_dir"])
    
    trainerargs = TrainingArguments(
        per_device_train_batch_size=trainer_config["per_device_train_batch_size"],
        gradient_accumulation_steps=trainer_config["gradient_accumulation_steps"],
        torch_empty_cache_steps=trainer_config["torch_empty_cache_steps"],
        lr_scheduler_type=trainer_config["lr_scheduler_type"],
        num_train_epochs=trainer_config["num_train_epochs"],
        run_name="run_{}".format(config["config_name"]),
        learning_rate=trainer_config["learning_rate"],
        logging_steps=trainer_config["logging_steps"],
        warmup_steps=trainer_config["warmup_steps"],
        weight_decay=trainer_config["weight_decay"],
        output_dir=trainer_config["output_dir"],
        max_steps=trainer_config["max_steps"],
        data_seed=trainer_config["seed"],
        optim=trainer_config["optim"],
        seed=trainer_config["seed"],
        report_to="none",
    )

    trainer = Trainer(
        tokenizer=llm.tokenizer,
        train_dataset=traindata,
        args=trainerargs,
        model=llm.model,
    )

    trainer.train()
    
    if os.path.isdir(trainer_config["output_dir"]):
        shutil.rmtree(trainer_config["output_dir"])

    save_path = os.path.join(trainer_config["save_dir"], "{}-{}".format(config["model_config"]["huggingface_model"], datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    llm.save(save_path)

    with open(os.path.join(save_path, "custom_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Model and config saved to : {save_path}")    




if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    classes=[
        "BYA-87", #1
        "BRA-69", #2
        "TU", #3
        "U", #4
        "F", #5
        "BGA", #6
        "BFA", #7
        "%-BYA-97", #8 
        "T-BRA", #9
        "%-TU", #10
        "%-BYA", #10
        "BYA", #11
        "BRA", #13
        "%-BRA", #14
    ]

    include_idx = [10, 11, 12, 13]
    include_classes = [classes[i] for i in include_idx]

    id2label = {i: c for i, c in enumerate(include_classes)}
    label2id = {c: i for i, c in enumerate(include_classes)}

    
    model = AutoModelForSequenceClassification.from_pretrained(
            "ltg/norbert3-large", 
            num_labels=len(include_classes), 
            id2label=id2label, 
            label2id=label2id,
            trust_remote_code=True,
        )
    
    tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-large")

    data = load_data()

    print(data)

    # texts = ["Example text 1", "Example text 2", "Example text 3"]
    # labels = [0, 1, 2]  # Example labels
    
    # encodings = preprocess(texts, labels, tokenizer)
    # dataset = CustomDataset(encodings)
    
    # # DataLoader
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # train(model, train_loader)

