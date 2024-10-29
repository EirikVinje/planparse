import datetime
import json
import os


def generate_config(
        model : str = "NorwAI/NorwAI-Mistral-7B-instruct",
        ):
    """
    Generate a config file for the planparse pipeline.
    """

    config = {}

    config["model_config"] = {
        "savepath": "./models/{}-{}".format(model.split("/")[-1], datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        "huggingface_model": model,
        "torch_dtype": "float16",
        "load_in_4bit": False,
        "load_in_8bit": True,
        "lora_dropout": 0.1,
        "lora_alpha": 16,
        "device": "cuda",
        "r": 8,
        }

    config["trainer_config"] = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "logging_steps": 100000,
        "output_dir": "./temp",
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 200,
    }

    return config