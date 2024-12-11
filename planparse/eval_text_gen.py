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
from load_data import preprocess_text

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
    parser.add_argument("--docs", type=str, default='data/1719/1719.txt', help='Path to data file(s)')
    parser.add_argument("--local_path", type=str, default='local_models/NorwAI/NorwAI-Mistral-7B-instruct-20241203104742')
    args = parser.parse_args()
    
    '''data = [
        "Elles kan frittståande bodar og garasjar inntil 50 kvm BYA plasserast som vist på Byggegrense 1 på plankart.",
        "Her er det ikke lov med mer enn BYA på 200 kvadratmeter.",
        "Det tillates maksimal BGA - 200 kvadratmeter.",
        "Her kan man bygge så mye som man vil.",
        "Lovlig bebygd areal BYA er 160%.",
        ]'''

    data = []
    paths = args.docs.split(',')
    for filepath in paths:
        filepath = filepath.strip(' ')
        file = open(filepath,'r')
        content = file.read()
        content = preprocess_text(content)
        data.append(content)
        file.close()
    #print(data)

    data = ["Det tillates maksimal % BYA på 50 %. Side 2 av 8 Det kreves et minste uteoppholdsareal MUA pr. boligenhet på 50 m2. Bebyggelsen skal ha saltak med vinkel mellom 18 og 38 grader. Det tillates påbygg i form av ark/oppløft, der påbyggets bredde kan være maksimalt  1/3-del av takets lengde. B2: I byggeområde B 2 tillates oppført tofamilie-/trefamilie-/firefamiliehus i maksimalt to  etasjer, med mønehøyde maksimalt 10,0 m over planert terreng og med mulighet for  beboelse på innredet loft, og med tilhørende anlegg som garasjer og separate boder.  Garasjer og boder skal tilpasses bolighusets form, takvinkel og materialbruk, de skal  kun ha en etasje og maksimal mønehøyde på 5,5 m og kan oppføres utenfor  byggegrense med avstand minst 2,0 m til nabogrense mot kommunal vei dersom  utkjøring fra garasje skjer parallelt med vei. Skjer utkjøring direkte mot kommunal vei  skal avstanden til nabogrense mot vei være minst 5,0 m. Byggegrense mot gang-/sykkelvei langs FV 304 er 15 m fra midtlinje av gang-  sykkelvei. Det tillates maksimal % BYA på 50 %. Byggeområde for industri 11 I byggeområde I 1 ligger Kvelde mølle med tilhørende lagerbebyggelse og tekniske  installasjoner. Maksimalt tillatt % BYA er på 50 %."]

    
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = generate_config()
    logger.info("loaded config from : {}".format(args.config))
    
    print()
    pprint(config, sort_dicts=False)
    print()
    
    llm = LLMBaseTextGen(config, args.local_path)
    llm.load_model()
    logger.info("loaded model : {}".format(config["model_config"]["huggingface_model"]))
    
    config["template_path"] = "./prompt_templates/mistral_7b_test_v5.jinja"
    config["generation_config"]["max_new_tokens"] = 300
    prompter = Prompter(config["template_path"])
    prompter.load()
    logger.info("loaded promp template from : {}".format(config["template_path"]))

    documents = [prompter(doc, None) for doc in data]
    print(documents[0])
    logger.info("prompts generated...")

    logger.info("starting generation...")
    outputs = llm.generate(documents, config["generation_config"])
    
    for output, raw in zip(outputs, data):
        print()
        print("IN: {}".format(raw))
        print("OUT: {}".format(output))
    print()



