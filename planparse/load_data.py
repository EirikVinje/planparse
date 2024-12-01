import os
import json
import glob
import pandas as pd

def create_text_row(instruction, input, output):
    text_row = f"<s>[INST] {instruction} {input} [/INST] \\n {output} </s>"
    return text_row

def process_jsonl_file(output_file_path, dataset):
    with open(output_file_path, "w") as output_jsonl_file:
        for _, row in dataset.iterrows():
            json_object = {
                "text": create_text_row(row["instruction"], row["input"], row["output"]),
                "instruction": row["instruction"],
                "input": row["input"],
                "output": str(row["output"])
            }
            output_jsonl_file.write(json.dumps(json_object) + "\n")

def load_data(data_folder):
    datafiles = glob.glob(f'{data_folder}/**/*.txt', recursive=True)
    y_files = glob.glob(f'{data_folder}/**/*.json', recursive=True)
    
    instruction = ("Du er en ekspert på byggeplansaker i kommunale arkiv. Din oppgave er å kjenne igjen om en tekst inneholder 'utnyttingsgrad', dette er en betegnelse for hvor mye en kan bygge på et bestemt område. Hvis det finnes tegn til utnyttingsgrad i dokumentet, må du spesifisere hvilke typer utnyttingsgrad som nevnes, ved å gjenkjenne følgende koder og mønstre: BYA er Bebygd areal, oppgitt i kvadratmeter (f.eks. 'BYA 50 kvm') eller i prosent (f.eks. 'BYA = 35%'). BRA er Bruksareal, oppgitt i kvadratmeter (f.eks. 'BRA 120 kvm') eller i prosent (f.eks. 'BRA = 120%'). %-BYA er Prosent bebygd areal (f.eks. 'BYA = 35%'). %-BRA er Prosent bruksareal (f.eks. 'BRA = 60%'). Gi kun svar som en Python-dictionary med formatet: {'utnyttingsgrad': ['BYA', 'BRA', '%-BYA', '%-BRA']}. Hvis ingen av de spesifiserte utnyttingsgradene finnes i teksten, returner en tom liste. Her er teksten: ")
    inst_list = [instruction]*len(datafiles)
    
    inputs = []
    outputs = []
    for filepath in datafiles:
        with open(filepath, 'r') as file:
            inputs.append(file.read())
    
    for file in y_files:
        with open(file, 'r') as f:
            outputs.append(json.load(f))
    max_length = max(len(inputs), len(outputs))
    inputs += [None] * (max_length - len(inputs))
    outputs += [None] * (max_length - len(outputs))
    dataset = pd.DataFrame({'instruction': inst_list, 'input': inputs, 'output': outputs})

    #print(dataset)
    process_jsonl_file(f'{data_folder}/training_dataset.jsonl', dataset)
    print("DONE")
