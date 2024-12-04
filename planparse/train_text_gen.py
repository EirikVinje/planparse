import argparse
import datetime
import shutil
import json
import os

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

from generate_config import generate_config

from llm_base_text_gen import LLMBaseTextGen
from load_data import load_data_prompter


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
        data_collator=DataCollatorForLanguageModeling(llm.tokenizer, mlm=False),
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


def jsonln_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)
            yield line


def load_data(file_path):

    examples = []

    for line in jsonln_loader(file_path):
        examples.append(line)

    return examples


def format_tokenize(data, tokenizer):
    
    data = [sample for sample in data["text"]]

    tokenized_data = tokenizer(data, truncation=True)
    
    return [{"input_ids": tokens_ids, "attention_mask": atm} 
            for tokens_ids, atm in zip(tokenized_data["input_ids"], tokenized_data["attention_mask"])]


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    if args.config is None:
        config = generate_config(f"config_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), round(torch.cuda.memory_reserved(0)/1024**3,1)))
    
    data = load_data_prompter(config["train_path"])
    
    llm = LLMBaseTextGen(config)
    llm.load_model()
    llm.init_training()
    llm.to_device()

    tokenized_data = format_tokenize(data, llm.tokenizer)

    train(
        llm,
        config, 
        tokenized_data,
        )

    
