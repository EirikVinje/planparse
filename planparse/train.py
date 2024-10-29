import argparse
import shutil
import json
import os

from transformers import Trainer, TrainingArguments
import torch
from generate_config import generate_config
from llm_base_text_gen import LLMBase

def train(
        config : dict, 
        traindata, 
        ):

    if os.path.isdir(config["output_dir"]):
        shutil.rmtree(config["output_dir"])

    llm = LLMBase(config)
    llm.load_model()
    llm.init_training()
    llm.to_device()
    
    trainerargs = TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        output_dir=config["output_dir"],
    )

    trainer = Trainer(
        tokenizer=llm.tokenizer,
        train_dataset=traindata,
        data_collator=None,
        args=trainerargs,
        model=llm.model,
    )

    trainer.train()
    
    if os.path.isdir(config["output_dir"]):
        shutil.rmtree(config["output_dir"])

    # llm.save()


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    if args.config is None:
        config = generate_config()
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))

    train(config["trainer_config"])




