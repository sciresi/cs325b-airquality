#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    experiment.py train --yaml-file=<yaml-file> [options]
    experiment.py evaluate --yaml-file=<yaml-file> [options]
    experiment.py test --model-path=<model-path> [--results-folder=<results-folder>]
Options:
    -h --help                               show this screen.
    --no-cuda                               don't use GPU (by default will try to use GPU)
    --log-file=<log-file>                   name of log file to log all output
"""

import os
from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import model_implementations
import model_types

import numpy as np

from sklearn import metrics
import matplotlib.pyplot as plt

from utils import read_yaml
from dataloader import load_data_new
from pdb import set_trace

def log(string, log_file = None):
    """
    A convenience function to print to standard output as well as write to
    a log file. Note that LOG_FILE is a global variable set in main and
    is specified by a command line argument. See docopt string at top
    of this file.
    
    Parameters
    ----------
    string : str 
        string to print and log to file
    """
    if log_file is not None:
        log_file.write(string + '\n')
    if LOG_FILE is not None: # by default, look for globally set log file
        LOG_FILE.write(string + '\n')
    print(string)

def run_training(model, train_data, num_epochs = None, 
                 val_data = None, results_folder = None):
    """
    Trains the model on the given dataset.
    
    Parameters
    ----------
    model : model_types.Model
        A Model instance (e.g., PyTorchModel) to train.
    train_data : training data
        Training data for the model. The type depends on the model type (e.g., 
        a PyTorchModel will be expecting a torch.utils.data.DataLoader).
    num_epochs : int, optional
        Number of epochs to train the model for. If None, the Model will train
        for a default number of epochs (specified in model_types.py).
    val_data : validation data, optional
        Optional validation data to evaluate the Model during training. Should
        be the same type as train_data if provided.
    results_folder : str, optional
        Optional path to a folder to save training results, such as final
        losses and a loss curve plot.
    """
    model.train(train_data, num_epochs, val_data)
    plot_save_path = None
    if results_folder:
        plot_save_path = os.path.join(results_folder, "loss_graph.png")
        model_save_path = os.path.join(results_folder, "trained_model")
        model.save(model_save_path)
    model.plot_loss_curve(plot_save_path)

# TODO: need to set other random seeds???
def set_random_seed(seed):
    """
    Seeds the random number generator for PyTorch for reproducibility.
    
    Parameters
    ----------
    seed : int or None
        Seed for PyTorch's random number. If None, a warning message is
        printed out.
    """
    if seed:
        log("Seeding PyTorch with random seed: {}".format(seed))
        torch.manual_seed(seed)
    else:
        log("WARNING: The random number generator has not been seeded.")
        log("You are encouraged to run with a random seed for reproducibility!")

# TODO: parameters other than learning rate
def create_optimizer(model, optimizer_name, lr, weight_decay):
    """
    Instantiates a torch.optim.Optimizer with the specified parameters.
    
    Parameters
    ----------
    model : model_types.PyTorchModel
        Model whose parameters the optimizer will tune.
    optimizer_name : str
        Name of the Optimizer to create (e.g., "Adam"). Should match the class
        name specified in torch.optim.
    lr : float
        Learning rate for the Optimizer.
        
    Returns
    -------
    optimizer : torch.optim.Optimizer
        Optimizer to use for training.
    """
    log("Loading optimizer {} with learning rate {}".format(optimizer_name, lr))
    return getattr(optim, optimizer_name)(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)

def make_results_folder(results_folder):
    """
    Checks if the results folder specified by the YAML file exists already. If
    if doesn't, it will be created. If it already exists but isn't a directory,
    a ValueError is thrown. If a results folder was not specified, a warning
    message is printed to emphasize that results will not be saved.
    
    Parameters
    ----------
    results_folder : str or None
        Path to folder to save results. If None or empty, a warning message is
        printed.
    """
    if results_folder:
        if not os.path.exists(results_folder):
            log("Creating folder at {} to save results...".format(results_folder))
            os.makedir(results_folder)
        elif not os.path.isdir(results_folder):
            raise ValueError("Specified results folder is already a regular file.")
        else:
            log("Will save results at {}...".format(results_folder))
    else:
        log("WARNING: Results will not be saved!")
        log("You are encouraged to specify a results folder in the YAML file and rerun.")    
    
def train(args, yaml_data):
    set_random_seed(yaml_data["seed"])
    make_results_folder(yaml_data["results_folder"])
    model = load_model(yaml_data)

    dataloaders = load_data_new(**yaml_data)
    train_dataloader = dataloaders["train"]
    val_dataloader = dataloaders.get("val")
    
    num_epochs = yaml_data.get("max_epoch")
    results_folder = yaml_data["results_folder"]
    losses = run_training(model, train_data = train_dataloader, val_data = val_dataloader, 
                          num_epochs = num_epochs, results_folder = results_folder)

def evaluate(model, val_iter, results_folder):
    model.eval()

    labels = []
    preds = []

    evals = []
    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

    
    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    mae = "MAE: {}".format(np.abs(evals - imputations).mean())
    mre = "MRE: {}".format(np.abs(evals - imputations).sum() / np.abs(evals).sum())
    nrmse = "NRMSE: {}".format(np.sqrt(np.power(evals - imputations, 2).mean()) / (evals.max() - evals.min()))

    with open(os.path.join(results_folder, "stats.txt"), 'w') as stats_file:
        log(mae, stats_file)
        log(mre, stats_file)
        log(nrmse, stats_file)

def load_model(yaml_data):
    model_type = getattr(model_types, yaml_data["model_type"])
    model_impl = getattr(model_implementations, yaml_data["model_impl"])
    
    # TODO: allow for parameters to pass to model_impl constructor?
    model = model_type(model_impl())
    if isinstance(model, model_types.PyTorchModel):
        lr = float(yaml_data["lr"])
        weight_decay = float(yaml_data.get("weight_decay", 0))
        optimizer = create_optimizer(model, yaml_data["optimizer"], lr, weight_decay)
        model.set_optimizer(optimizer)
    
    return model

def prep_eval(yaml_data, no_cuda):
    data_iter, model = load_data_and_model(yaml_data, "val_data")
    results_folder = yaml_data["results_folder"]
    model_save_path = os.path.join(results_folder, "trained_model.pt")
    if not no_cuda and torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    map_location = device if device_name == "cpu" else None
    model.load_state_dict(torch.load(model_save_path, map_location = map_location))
    if device_name == "cuda": 
        model.to(device)
    return data_iter, model

def set_log_file(log_file, results_folder):
    """
    Opens a writeable file object to log output and sets as a global variable. 
    
    Parameters
    ----------
    log_file : str
        Name of log file to write to, can be None
            * if None, no log file is created
    results_folder : str
        Folder to save log file to
    """
    global LOG_FILE
    LOG_FILE = None
    if log_file is not None:
        log_file_path = os.path.join(results_folder, log_file + ".txt")
        LOG_FILE = open(log_file_path, 'w')
        log("Created log file at {}".format(log_file_path))

def main():
    args = docopt(__doc__)
    yaml_data = read_yaml(args["--yaml-file"])
    set_log_file(args["--log-file"], yaml_data["results_folder"])
    if args["train"]:
        train(args, yaml_data)
    elif args["evaluate"]:
        data_iter, model = prep_eval(yaml_data, args["--no-cuda"])
        evaluate(model, data_iter, yaml_data["results_folder"])

    if LOG_FILE is not None:
        LOG_FILE.close()
        

if __name__ == '__main__':
    main()