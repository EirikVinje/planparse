import argparse
import shutil
import json
import os

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from generate_config import generate_config
from llm_base_text_gen import LLMBaseTextGen
from load_data import load_data
from datasets import load_dataset

def preprocess_function(examples, tokenizer):
    """
    Tokenize the inputs and align the labels for training.
    """
    model_inputs = tokenizer(
        examples["text"],  # Input text field
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="max_length",  # Ensures consistent input length
    )

    # Process labels (assuming 'label' is the field in the dataset)
    if "output" in examples:
        with tokenizer.as_target_tokenizer():  # For sequence-to-sequence tasks
            labels = tokenizer(
                examples["output"],  # Label field
                max_length=tokenizer.model_max_length,
                truncation=True,
                padding="max_length",
            )
            model_inputs["labels"] = labels["input_ids"]
    
    # Remove empty or malformed labels
    if "labels" in model_inputs and all(len(l) > 0 for l in model_inputs["labels"]):
        return model_inputs
    else:
        return None  # Drop samples with no valid labels

def train(config: dict, model_config: dict, traindata):
    """
    Train the model using the provided configurations and data.
    """
    if os.path.isdir(config["output_dir"]):
        shutil.rmtree(config["output_dir"])

    # Initialize the model
    llm = LLMBaseTextGen(model_config)
    llm.load_model()
    llm.init_training()
    llm.to_device()

    # Preprocess the training data
    traindata = traindata.map(
        lambda x: preprocess_function(x, llm.tokenizer), batched=True
    ).filter(lambda x: x is not None)  # Remove invalid samples

    # Use a data collator (optional, but recommended)
    data_collator = DataCollatorForSeq2Seq(tokenizer=llm.tokenizer, model=llm.model)

    # Training arguments
    trainerargs = TrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        output_dir=config["output_dir"],
        save_strategy="epoch",  # Save the model after each epoch
    )

    # Trainer
    trainer = Trainer(
        train_dataset=traindata,
        data_collator=data_collator,
        args=trainerargs,
        model=llm.model,
    )

    trainer.train()


if __name__ == "__main__":
    # Ensure the script runs from the project root
    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    if args.config is None:
        config = generate_config()
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

    # Display GPU info if available
    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print(
            "Memory Usage: {}/{}".format(
                round(torch.cuda.memory_allocated(0) / 1024**3, 1),
                round(torch.cuda.memory_reserved(0) / 1024**3, 1),
            )
        )

    # Load dataset
    cwd = os.getcwd()
    traindata = load_dataset("json", data_files="data/training_dataset.jsonl", split="train")

    # Train the model
    train(config["trainer_config"], config["model_config"], traindata)
