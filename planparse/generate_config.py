import argparse
import datetime
import json
import os


def generate_config(
        template_path : str = "./prompt_templates/mistral_7b_v1.jinja",
        model : str = "NorwAI/NorwAI-Mistral-7B-instruct",
        access_token : str = "./access_token/token.txt",
        ):
    """
    Generate a config file for the planparse pipeline.
    """

    config = {}    

    config["template_path"] = template_path

    config["model_config"] = {
        "savepath": "./models/{}-{}".format(model.split("/")[-1], datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        "access_token": access_token,
        "huggingface_model": model,
        "torch_dtype": "float16",
        "load_in_4bit": True,
        "load_in_8bit": False,
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

    config["generation_config"] = {
        "max_new_tokens": 30,
        "do_sample": False,
    }

    return config


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name of the config file")
    args = parser.parse_args()

    config = generate_config()

    savedir = "./configs"

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    savepath = os.path.join(savedir, args.name)

    with open(f"{savepath}.json", "w") as f:
        json.dump(config, f, indent=4)

    