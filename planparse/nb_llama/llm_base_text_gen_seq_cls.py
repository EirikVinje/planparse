from typing import List, Dict, Any
import json
import os

from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from huggingface_hub import login
import torch.nn as nn
import numpy as np
import torch




class CustomDataset(Dataset):
    def __init__(self, encodings, labels):

        self.encodings = []

        for i in range(len(encodings["input_ids"])):

            self.encodings.append(
                {
                    "input_ids" : encodings["input_ids"][i],
                    "attention_mask" : encodings["attention_mask"][i],
                    "labels" : labels[i],
                }
            )
        
        self.encodings = sorted(self.encodings, key=lambda x: len(x["input_ids"]), reverse=False)
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


class CausalTextClSModel(nn.Module):

    def __init__(
        self, 
        config : Dict, 
        local_path : str = None, 
        n_labels : int = None,
        ):

        super(CausalTextClSModel, self).__init__()
        
        self._config = config["model_config"]

        if config["access_token"] is not None or config["access_token"] != "none":
            login(config["access_token"])
        
        huggingface_model = self._config["huggingface_model"]
        load_in_4bit = self._config["load_in_4bit"]
        load_in_8bit = self._config["load_in_8bit"]
        lora_dropout = self._config["lora_dropout"]
        lora_alpha = self._config["lora_alpha"]
        r = self._config["r"]

        self.device = self._config["model_config"]["device"]
        self.local_path = local_path
        self.n_labels = n_labels
        
        self.backend = None
        self.tokenizer = None
        self.classifier_head = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=huggingface_model,
            padding_side="left",
        )

        self.tokenizer.add_special_tokens({'pad_token': "[PAD]"})
        
        # quantization_config = BitsAndBytesConfig(
        #         load_in_4bit=load_in_4bit,
        #         load_in_8bit=load_in_8bit,
        #     )

        self.backend = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=huggingface_model,
            # quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        
        self.backend.config.output_hidden_states = True
        self.backend.config.tie_word_embeddings = False

        print(f"Loaded base model from {huggingface_model}")

        self.lora_config = LoraConfig(
                target_modules=["q_proj", "k_proj", "v_proj", "dense_h_to_4h", "dense_4h_to_h"],
                lora_dropout=lora_dropout,
                lora_alpha=lora_alpha,
                task_type="CAUSAL_LM",
                bias="none",
                r=r,
            )

        self.peftmodel = get_peft_model(self.backend, self.lora_config)

        self.classifier_head = nn.Linear(self.backend.config.hidden_size, self.n_labels)

        # self.classifier_head.weight.data = self.classifier_head.weight.data.to(torch.float16)
        # self.classifier_head.bias.data = self.classifier_head.bias.data.to(torch.float16)

        print(f"Initialized classifier head with {self.n_labels} labels")

            

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        
        text_gen_output = self.peftmodel(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = text_gen_output.hidden_states[-1]
                
        pooled_output = hidden_states[:, -1, :]  
        
        logits = self.classifier_head(pooled_output)  

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
        
        output = SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

        return output
    

    def to_device(self, device=None):
        
        if device is None:
            device = self.device

        self.backend.to(device)
        self.classifier_head.to(device)


    def train(self):

        self.backend.train()
        self.classifier_head.train()
    

    def eval(self):

        self.backend.eval()
        self.classifier_head.eval()


if __name__ == "__main__":

    with open("configs/llama_1b.json", "r") as f:
        config = json.load(f)

    model = CausalTextClSModel(config, n_labels=2)

    train_x = [
        "Prosent",
        ]
    
    train_y = [0]

    tokenized_train_x = model.tokenizer(train_x, truncation=True, padding=False)
    train_dataset = CustomDataset(tokenized_train_x, train_y)

    trainer = Trainer(
        
        model=model,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        
        args=TrainingArguments(
            output_dir="./results",
            save_strategy="no",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            logging_steps=1,
            warmup_steps=0,
            weight_decay=0.01,
            eval_steps=1,
        ),
    )

    trainer.train()