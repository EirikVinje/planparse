import argparse
import json
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import torch


def infinite_input_inference(model, tokenizer):

    model.eval()

    device = "cuda"
    
    model.to(device)
        
    all_preds = []
    
    while True:

        input_text = input("skriv en tekst som omhandler utnyttingsgrad:\n")

        tokenized_input = tokenizer([input_text], return_tensors="pt")
        
        with torch.no_grad():
            
            logits = model(
                tokenized_input["input_ids"].to(device), 
                tokenized_input["attention_mask"].to(device)).logits
            
        pred = torch.argmax(logits, dim=1)
        print(pred)
        
        print("predicted 'utnyttingsgrad' : {}".format(model.config.id2label[pred[0].item()]))
        print()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to model")
    args = parser.parse_args()

    if not os.path.isfile("./setup.sh"):
        raise Exception("Please run from the root of the repository")
    
    if torch.cuda.is_available():
        print("Device: {}".format(torch.cuda.get_device_name(0)))
        print("Memory Usage: {}/{}".format(round(torch.cuda.memory_allocated(0)/1024**3,1), 
                                           round(torch.cuda.memory_reserved(0)/1024**3,1)))

    modelname = args.model

    tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True) 
    
    model = AutoModelForSequenceClassification.from_pretrained(
        modelname, 
        trust_remote_code=True,
    )
    
    infinite_input_inference(model, tokenizer)
    






