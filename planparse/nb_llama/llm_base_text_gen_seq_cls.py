from typing import List, Dict, Any
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PretrainedConfig

from transformers.modeling_outputs import SequenceClassifierOutput

import torch.nn as nn
import numpy as np
import torch


class CausalTextClSModel(nn.Module):

    def __init__(
        self, 
        config : Dict, 
        local_path : str = None, 
        n_labels : int = None,
        ):

        super(CausalTextClSModel, self).__init__()
        
        self.config = config
        self.local_path = local_path
        self.n_labels = n_labels
        
        self.model = None
        self.tokenizer = None
        self.classifier_head = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_config"]["huggingface_model"],
            padding_side="left",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.config["model_config"]["huggingface_model"],
        )

        print(f"Loaded base model from {self.config['model_config']['huggingface_model']}")

        self.model.config.output_hidden_states = True
        
        self.classifier_head = nn.Linear(self.model.config.hidden_size, self.n_labels)    
    

    def forward(self, input_ids, attention_mask=None, labels=None):
        
        text_gen_output = self.model(input_ids, attention_mask)
    
        hidden_states = text_gen_output.hidden_states[-1]
    
        pooled_output = hidden_states[:, -1, :]
    
        logits = self.classifier_head(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))


        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    

    def to(self, device):
        
        self.model.to(device)
        self.classifier_head.to(device)
        self.config["model_config"]["device"] = device


    def train(self, mode=True):
        
        super().train(mode)
        self.model.train(mode)
        self.classifier_head.train(mode)


    def eval(self):
        
        super().eval()
        self.model.eval()
        self.classifier_head.eval()


    def parameters(self, recurse: bool = True):
        return list(self.model.parameters()) + list(self.classifier_head.parameters())


    def save_pretrained(self, save_directory):
        """
        Save the model, tokenizer, and additional components to the specified directory.
        """
        # Save the base model and tokenizer
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save classifier head if it exists
        if self.model_type == "seq_cls" and self.classifier_head:
            
            classifier_state = {
                "state_dict": self.classifier_head.state_dict(),
                "hidden_size": self.model.config.hidden_size,
                "n_labels": self.n_labels,
            }
            
            torch.save(classifier_state, f"{save_directory}/classifier_head.pt")

        # Save configuration
        with open(f"{save_directory}/model_config.json", "w") as f:
            json.dump(self.config, f)
    

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Load a model, tokenizer, and additional components from the specified directory.
        """
        # Load configuration
        with open(f"{load_directory}/model_config.json", "r") as f:
            config = json.load(f)
        
        # Determine model type
        classifier_head_path = f"{load_directory}/classifier_head.pt"
        n_labels = None
        if os.path.exists(classifier_head_path):
            classifier_state = torch.load(classifier_head_path)
            n_labels = classifier_state["n_labels"]
        
        # Initialize model
        instance = cls(config=config, local_path=load_directory, n_labels=n_labels)
        
        # Load classifier head if it exists
        if n_labels is not None:
            instance.classifier_head.load_state_dict(classifier_state["state_dict"])
        
        return instance
    
    