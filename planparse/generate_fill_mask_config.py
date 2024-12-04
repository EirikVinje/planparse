import argparse
import datetime
import json
import os


def generate_config(
        config_name : str,
        model : str = "ltg/norbert3-large",
        ):
    """
    Generate a config file for the planparse pipeline.
    """

    config = {}    

    config["config_name"] = config_name
    config["train_path"] = "./data/training_dataset.jsonl"

    config["model_config"] = {
        "huggingface_model": model,
        "device": "cuda",
        }
    
    config["trainer_config"] = {
        "save_dir": "./local_models/",
        "run_name" : "run_{}".format(config["config_name"]),
        "per_device_train_batch_size" : 1,
        "gradient_accumulation_steps" : 1,
        "torch_empty_cache_steps" : True,
        "lr_scheduler_type" : "linear",
        "logging_steps" : 100000,
        "output_dir" :  "./temp",
        "optim" : "adamw_torch",
        "num_train_epochs" : 1,
        "learning_rate" : 1e-4,
        "warmup_steps" : 200,
        "weight_decay" : 0.01,
        "max_steps" :   -1,
        "report_to" : "none",
        "data_seed" : 42,
        "seed" : 42,
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

    