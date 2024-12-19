import tikzplotly
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_eval_acc(file):
    # read csv file as pd dataframe
    df = pd.read_csv(file)
    # plot the eval_accuracy, only plot the numbers, not N/A
    df['eval_accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    eval_data = df.dropna(subset=['accuracy'])

    eval_epochs = df.loc[eval_data.index - 1, 'Epoch'].values
    eval_accuracy = eval_data['accuracy'].values
    return eval_epochs, eval_accuracy

def plot_training_loss(file):
    df = pd.read_csv(file)
    # remove all N/A rows
    df = df.dropna(subset=['Loss'])
    # plot the training loss
    train_epochs = df['Epoch'].values
    train_loss = df['Loss'].values
    smoothed_accuracy = pd.Series(train_loss).rolling(window=15, center=True).mean()
    return train_epochs, smoothed_accuracy


def plot_all(result_files, result_names, save_folder):

    # plot all the eval_accuracy lines with plotly
    fig = go.Figure()
    for file in result_files:
        eval_epochs, eval_accuracy = plot_eval_acc(file)
        fig.add_trace(go.Scatter(x=eval_epochs, y=eval_accuracy, mode='lines', name=str((file.split('/')[4]).split('_')[2])))
    
    fig.update_layout(title='Evaluation accuracies', xaxis_title='Epoch', yaxis_title='Accuracy')
    fig.write_image(f"{save_folder}/eval_accuracies.png")
    #fig.show()
    # save to tikz file
    tikzplotly.save(f"{save_folder}/eval_accuracies.tex", fig)

    # trianing loss
    fig1 = go.Figure()
    for file in result_files:
        train_epochs, train_loss = plot_training_loss(file)

        fig1.add_trace(go.Scatter(x=train_epochs, y=train_loss, mode='lines', name=str((file.split('/')[4]).split('_')[2])))
    
    fig1.update_layout(title='Training losses', xaxis_title='Epoch', yaxis_title='Loss')
    fig1.write_image(f"{save_folder}/training_losses.png")
    #fig1.show()
    # save to tikz file
    tikzplotly.save(f"{save_folder}/training_losses.tex", fig1)

def plot_bar_plot(result_files, result_names, save_folder):
    fig1 = go.Figure()
    train_runtimes = []
    names = []
    for file in result_files:
        # read json files
        #print(file)
        with open(file) as f:
            data = json.load(f)
        train_runtime = data['train_runtime']
        train_runtimes.append(train_runtime)
        names.append(str((file.split('/')[4]).split('_')[1]))
    
    # sort the runtimes
    train_runtimes, names = zip(*sorted(zip(train_runtimes, names)))
    print(train_runtimes, names)
    fig1.add_trace(go.Bar(x=names, y=train_runtimes, name='Training runtime'))

    fig1.update_layout(title='Training runtimes', xaxis_title='Model', yaxis_title='Runtime')
    fig1.write_image(f"{save_folder}/training_runtimes.png")
    #fig1.show()
    # save to tikz file
    #tikzplotly.save(f"{save_folder}/training_runtimes.tex", fig1)

    



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, help="Path to the run folder", default="./training_logs/relevant")
    parser.add_argument("--save_folder", type=str, help="Folder to save the plots", default="./plots/")
    args = parser.parse_args()
    run_path = args.run_path
    save_folder = args.save_folder


    result_files = glob.glob(f"{run_path}/*/N4*/*.csv")
    print("These are the result files that will be plotted: \n", result_files)
    result_names = [str((file.split('/')[4]).split('_')[1]) for file in result_files]
    print("These are the result names: \n", result_names)

    plot_all(result_files, result_names, save_folder)

    result_files_json = glob.glob(f"{run_path}/*/N4*/results.json")
    plot_bar_plot(result_files_json, result_names, save_folder)

    result_files = glob.glob(f"{run_path}/bert/N4*/*.csv")
    result_names = [str((file.split('/')[4]).split('_')[1]) for file in result_files]
    save_folder_bert = save_folder + "bert"
    plot_all(result_files, result_names, save_folder_bert)

    result_files = glob.glob(f"{run_path}/generative/N4*/*.csv")
    result_names = [str((file.split('/')[4]).split('_')[1]) for file in result_files]
    save_folder_generative = save_folder + "generative"
    plot_all(result_files, result_names, save_folder_generative)