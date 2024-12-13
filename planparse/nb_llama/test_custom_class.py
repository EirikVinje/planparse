from typing import List, Dict, Any
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Trainer, TrainingArguments

from transformers.modeling_outputs import SequenceClassifierOutput


from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import torch

from planparse.prompter import Prompter


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


class CausalTextClSModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(
        self,
        huggingface_model: str = None,
        local_path: str = None,
        n_labels: int = None,
        device: str = "cuda",
    ):  
        
        config = {
        "_name_or_path": "meta-llama/Llama-3.2-1B",
        "architectures": [
            "LlamaForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        "rope_theta": 500000.0,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.46.3",
        "use_cache": True,
        "vocab_size": 128256
        }

        config = PretrainedConfig.from_dict(config)

        super().__init__(config)

        self.huggingface_model = huggingface_model
        self.local_path = local_path
        self.n_labels = n_labels

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.huggingface_model,
            padding_side="left",
        )

        self._base_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.huggingface_model,
        )

        print(f"Loaded base model from {self.huggingface_model}")

        self.classifier_head = nn.Linear(self._base_model.config.hidden_size, self.n_labels)


    def forward(self, input_ids=None, attention_mask=None, labels=None):
        
        text_gen_output = self._base_model(input_ids, attention_mask)

        hidden_states = text_gen_output.last_hidden_state

        pooled_output = hidden_states[:, -1, :]  # CLS token representation

        logits = self.classifier_head(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
        
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save base model and tokenizer
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save classifier head
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
        # Load configuration
        with open(f"{load_directory}/model_config.json", "r") as f:
            config = json.load(f)

        # Load classifier head
        classifier_head_path = f"{load_directory}/classifier_head.pt"
        n_labels = None
        if os.path.exists(classifier_head_path):
            classifier_state = torch.load(classifier_head_path)
            n_labels = classifier_state["n_labels"]

        # Initialize model
        instance = cls(config=config, local_path=load_directory, n_labels=n_labels)

        # Load classifier head if exists
        if n_labels is not None:
            instance.classifier_head.load_state_dict(classifier_state["state_dict"])

        return instance




if __name__ == "__main__":

    with open("configs/llama_1b.json", "r") as f:
        config = json.load(f)

    model = CausalTextClSModel(
        huggingface_model="NbAiLab/nb-llama-3.2-1B-Instruct", 
        n_labels=5
        )

    # input_ids = torch.tensor([[1, 2, 10, 4]]).to("cuda")
    # attention_mask = torch.tensor([[1, 1, 1, 1]]).to("cuda")
    # labels = torch.tensor([0]).to("cuda")

    # output = model.forward(input_ids, attention_mask, labels)

    train_x = [
        "Prosent bebygd areal er 27%.",
        "bruksareal er 30 kvm."
        ]
    
    train_y = [0, 1]

    tokenized_train_x = model.tokenizer(train_x, truncation=True, padding=False)
    train_dataset = CustomDataset(tokenized_train_x, train_y)

    trainer = Trainer(
        
        model=model,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        
        args=TrainingArguments(
            output_dir="./results",
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