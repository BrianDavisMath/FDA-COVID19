import torch
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import time

from modules.model import Model
from modules.data import get_data
from feature_selection.feature_selection import RandomForestFeatureSelection, SparsePCAFeatureSelection, GeneticFeatureSelection
from utils import get_parser


parser = get_parser()
args = parser.parse_args()

device = "cuda"
# n_epochs = 20
# batch_size = 3
# display_step = 1
# feature_selector = 'none'

if args.fs == "random_forest":
    feature_selector = RandomForestFeatureSelection()
    feature_selector.load("location_of_saved_random_forest.pkl")
elif args.fs == "sparse_pca":
    feature_selector = SparsePCAFeatureSelection()
elif args.fs == "none":
    feature_selector = None
    
train_data, valid_data = get_data(args.data_path, feature_selector)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)



# default inoput size is 1000
model = Model(dim=args.hidden_dim, n_layers=args.n_layers, in_size=17).to(device)

opt = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

def validation(loader, model, loss_fn):
    accuracy = 0
    losses = 0
    weighted_losses = 0
    n_total = 0

    for x, y, weights in valid_loader:
        x, y, weights = x.to(device), y.to(device), weights.to(device)
        
        logits = model(x)

        unweighted_loss = loss_fn(logits, y)
        weighted_loss = unweighted_loss*weights

        losses += unweighted_loss.sum().item()
        weighted_losses += weighted_loss.sum().item()
        accuracy += ((logits > 0) == y).sum().item()
        n_total += x.shape[0]
    
    return accuracy/n_total, losses/n_total, weighted_losses/n_total

total_iters = 0
for epoch in range(1, args.epochs+1):
    epoch_time = time.time()
    print(f"epoch {epoch:02d}\n")
    for iter, (x, y, weights) in enumerate(train_loader):
        x, y, weights = x.to(device), y.to(device), weights.to(device)
        logits = model(x)

        unweighted_loss = loss_fn(logits, y)
        weighted_loss = unweighted_loss*weights
        loss = weighted_loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_iters += 1
        if not total_iters % args.display_step:
            unweighted_tr_loss = unweighted_loss.mean().item()
            weighted_tr_loss = weighted_loss.mean().item()
            val_accuracy, unweighted_val_loss, weighted_val_loss = validation(valid_loader, model, loss_fn)

            print(epoch, iter, unweighted_tr_loss, weighted_tr_loss, unweighted_val_loss, weighted_val_loss, val_accuracy)
        
    print(f"time: {time.time() - epoch_time:.2f}s")

"""
todo:
    feature selector arguments CL option
    feature selection utils function
    unweighted / weighted loss blend
    use parser
    dataparallel
"""