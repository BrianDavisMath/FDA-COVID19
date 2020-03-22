import torch
from torch.utils.data import DataLoader

from modules.model import Model
from modules.data import BioactivityData
from modules.feature_selectors import ExampleFeatureSelector

device = "cpu"
raw_data_path = None
processed_data_path = None
n_epochs = 10
feature_selector = ExampleFeatureSelector()
train_data = BioactivityData(raw_data_path, processed_data_path, feature_selector)
train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

valid_data = None
valid_loader = None

input_size = train_data[0][0].shape[0]
model = Model(input_size, dim=200, n_res_blocks=2).to(device)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

def training_epoch(loader, model, opt, loss_fn):
    for iter, (x, y) in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

def validation(loader, model, loss_fn):
    accuracy = 0
    losses = 0
    n_total = 0

    for iter, (x, y) in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)

        accuracy += (pred == y).sum().item()
        loss += x.shape[0]*loss_fn(pred, y).item()
        n_total += x.shape[0]
    
    return accuracy/n_total, loss/n_total


    
for epoch in range(1, n_epochs+1):
    # train for an epoch
    print(f"epoch {epoch:02d}\n")
    training_epoch(train_loader, model, optimizer, loss_fn)

    # print validation accuracy & loss
    valid_accuracy, valid_loss = validation(valid_loader, model, loss_fn)

    print(f"valid accuracy: {valid_accuracy:.3f}   valid loss: {valid_loss:.4f}\n\n")

    

    # print validation score

"""
todo:
    parser
    cuda
    dataparallel
"""