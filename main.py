from transformers import BertModel
import time, os
import torch
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

from typing import List, Mapping, Optional
Outputs = Mapping[str, List[torch.Tensor]]

import matplotlib.pyplot as plt
from model import *
from loss import *
from train import *
from evaluation import *
from data import *
from caliberation_plots import *
    
def main_fxn():
    
    parser = argparse.ArgumentParser('')
    parser.add_argument('-gpu', type = int, default = 1)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type = int, default= 128)
    parser.add_argument('-batch_size', type = int, default= 32)
    parser.add_argument('-lr', type = float, default = 2e-5)
    parser.add_argument('-print_every', type = int, default= 100)
    parser.add_argument('-max_eps', type = int, default= 1)
    parser.add_argument('-soft_labels', type = bool, default= True)
    parser.add_argument('-experiment_root', type = str, default= './Models/Soft_Label_Training/')
    parser.add_argument('-train_file', type=str, default= './data/train_data.csv')
    parser.add_argument('-val_file', type=str, default= './data/val_data.csv')
    args = parser.parse_args('')

    if not os.path.exists(args.experiment_root):
        os.makedirs(args.experiment_root)
    
    #Instantiating the classifier model
    print("Building model! (This might take time if you are running this for first time)")
    st = time.time()
    net = Classifier(args.freeze_bert)
    net.cuda(args.gpu) #Enable gpu support for the model
    print("Done in {} seconds".format(time.time() - st))
    
    
    print("Creating criterion and optimizer objects")
    st = time.time()
    if(args.soft_labels):
        criterion = cross_entropy_with_probs
    else:
        criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(net.parameters(), lr = args.lr)
    print("Done in {} seconds".format(time.time() - st))
    
    
    #Creating dataloaders
    print("Creating train and val dataloaders")
    st = time.time()
    train_set = FactualityDataset(filename = args.train_file, maxlen = args.maxlen)
    val_set = FactualityDataset(filename = args.val_file, maxlen = args.maxlen)
    train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 5)
    val_loader = DataLoader(val_set, batch_size = args.batch_size, num_workers = 5)
    print("Done in {} seconds".format(time.time() - st))
    
    
    print("Let the training begin")
    st = time.time()
    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))
    
    
    # Evaluate on Best Saved Model
    best_model_path = os.path.join(args.experiment_root, 'best_model.pth')
    net = Classifier(args.freeze_bert)
    net = load_ckp(best_model_path, net, args)

    mean_acc, mean_loss, preds, pred_probs, true_labels, prob_labels,\
    sentences, losses_per_sample = evaluate(net, criterion, val_loader, args)

    
    preds = np.concatenate(preds)
    pred_probs = np.concatenate(pred_probs)
    losses_per_sample = np.concatenate(losses_per_sample)

    true_labels = np.concatenate(true_labels)
    prob_labels = np.concatenate(prob_labels)

    sentences = np.concatenate(sentences)
    
    q = pred_probs.copy()
    y = prob_labels.copy()
    
    do_caliberation(q, y)
    
    print("completed")
    
if __name__ == "__main__":
    main_fxn()
    