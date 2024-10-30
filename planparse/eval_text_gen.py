from pprint import pprint
import argparse
import logging
import json
import time
import os

from llm_base_fill_mask import LLMBaseFillMask
from llm_base_text_gen import LLMBaseTextGen
from generate_config import generate_config
from prompter import Prompter
from read_pdf import read_pdf

if os.path.isfile("./setup.sh") is False:
    raise RuntimeError("This script must be run from the root of the repository.")

logger = logging.getLogger("planparse")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()        
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def evaluate():
    
    pass


if __name__ == "__main__":

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/mistral7b.json", help="Path to config file")
    args = parser.parse_args()
    
    data = [
        "Elles kan frittståande bodar og garasjar inntil 50 kvm BYA plasserast som vist på Byggegrense 1 på plankart.",
        "Her er det ikke lov med mer enn BYA på 200 kvadratmeter.",
        "Det tillates maksimal BGA - 200 kvadratmeter.",
        "Her kan man bygge så mye som man vil.",
        "Lovlig bebygd areal BYA er 160%.",
        ]
    
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = generate_config()
    logger.info("loaded config from : {}".format(args.config))
    
    print()
    pprint(config, sort_dicts=False)
    print()
    
    llm = LLMBaseTextGen(config["model_config"])
    llm.load_model()
    logger.info("loaded model : {}".format(config["model_config"]["huggingface_model"]))
    
    config["template_path"] = "./prompt_templates/mistral_7b_v4.jinja"
    prompter = Prompter(config["template_path"])
    prompter.load()
    logger.info("loaded promp template from : {}".format(config["template_path"]))    

    documents = [prompter(doc) for doc in data]
    logger.info("prompts generated...")

    logger.info("starting generation...")
    outputs = llm.generate(documents, config["generation_config"])
    
    for output, raw in zip(outputs, data):
        print()
        print("IN: {}".format(raw))
        print("OUT: {}".format(output))
    print()



