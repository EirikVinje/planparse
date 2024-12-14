import argparse
import datetime
import json
import os


def generate_config(
        config_name : str,
        template_path : str = "./prompt_templates/mistral_7b_v1.jinja",
        model : str = "NorwAI/NorwAI-Mistral-7B-instruct",
        access_token : str = "./access_token/token.txt",
        ):
    """
    Generate a config file for the planparse pipeline.
    """

    config = {}    

    config["config_name"] = config_name
    config["template_path"] = template_path
    config["train_path"] = "./formated_data/huge/splits/train.jsonl"
    config["eval_path"] = "./formated_data/huge/splits/eval.jsonl"
    config["test_path"] = "./formated_data/huge/splits/test.jsonl"

    config["model_config"] = {
        "access_token": None,
        "huggingface_model": model,
        "torch_dtype": "float32",
        "load_in_4bit": False,
        "load_in_8bit": False,
        "lora_dropout": 0.1,
        "lora_alpha": 16,
        "device": "cuda",
        "r": 8,
        }

    
    run_name = model.split("/")
    run_name = "_".join(run_name).replace(".", "_")
    
    config["trainer_config"] = {
        "run_name": run_name,
        "lr_scheduler_type": "linear",
        "save_dir": "./local_models/",
        "output_dir": "./temp",
        "optim": "adamw_torch",
        "eval_strategy": "epoch",
        "num_train_epochs": 1,
        "save_strategy": "no",
        "report_to": "none",
        "logging_steps": 1,
        "torch_empty_cache_steps": True,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "warmup_steps": 200,
        "weight_decay": 0.001,
        "max_steps": -1,
        "data_seed": 42,
        "seed": 42
    }

    config["label2id"] = {
        "none" : 0,
        "%-BYA" : 1,
        "BYA" : 2,
        "BRA" : 3,
        "%-BRA" : 4
    }

    return config


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name of the config file")
    args = parser.parse_args()

    configname = args.name

    config = generate_config(config_name=configname)

    savedir = "./configs"

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    savepath = os.path.join(savedir, configname)

    with open(f"{savepath}.json", "w") as f:
        json.dump(config, f, indent=4)

    