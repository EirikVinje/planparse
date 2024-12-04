import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


def main():

    tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-large")
    model = AutoModelForMaskedLM.from_pretrained("ltg/norbert3-large", trust_remote_code=True)
    
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")

    text = "Denne eiendommen har bruker utnyttingsgrad av type[MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    
    output = model(**inputs)

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = output.logits[0, mask_token_index, :]
    
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
    
    for token in top_5_tokens:
         print(f"'{text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")
        

if __name__ == "__main__":
    main()