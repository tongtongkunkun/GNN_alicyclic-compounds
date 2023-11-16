import csv

import dgllife.utils
from rdkit import Chem
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from dgllife.model import WeavePredictor
import dgl
from torch.optim.lr_scheduler import ReduceLROnPlateau

data = pd.read_csv('demo7.csv')
data['output'].replace({'Inactive': 0, 'Active': 1}, inplace=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_data, test_data, train_label, test_label = train_test_split(data['smiles'], data['output'], test_size=0.3, random_state=0)
train_data = train_data.reset_index(drop=True)
train_label = train_label.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
test_label = test_label.reset_index(drop=True)
train_mol = []
test_mol = []
for smiles in train_data:
    mol = Chem.MolFromSmiles(smiles)
    train_mol.append(mol)
for smiles in test_data:
    mol = Chem.MolFromSmiles(smiles)
    test_mol.append(mol)

atom_featurizer = dgllife.utils.CanonicalAtomFeaturizer()
bond_featurizer = dgllife.utils.CanonicalBondFeaturizer()

train_g = []
for mol in train_mol:
    graph = dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    train_g.append(graph)

test_g = []
for mol in test_mol:
    graph = dgllife.utils.mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    test_g.append(graph)

train_dataset = list(zip(train_g, train_label))
test_dataset = list(zip(test_g, test_label))

train_loader = GraphDataLoader(train_dataset, batch_size=64)
test_loader = GraphDataLoader(test_dataset,  batch_size=64)
lr = 1e-3
num_epochs = 100

model = WeavePredictor(node_in_feats=74, edge_in_feats=12, gnn_hidden_feats=64, graph_feats=64, num_gnn_layers=3)

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
train_losses = []
test_losses = []
train_roc_aucs = []
test_roc_aucs = []
best_accuracy = 0
best_model_state_dict = None

for epoch in range(num_epochs):
    # Train the model
    model.train()
    epoch_loss = 0
    y_true = []
    y_pred = []
    for batch_id, batch_data in enumerate(train_loader):
        bg, labels = batch_data
        bg = bg.to(device)
        labels = labels.to(device)
        n_feats = bg.ndata.pop('h')
        e_feats = bg.edata.pop('e')
        n_feats = n_feats.to(device)
        e_feats = e_feats.to(device)
        labels = labels.to(torch.float32)
        logits = model(bg, n_feats, e_feats)
        loss = loss_fn(logits.view(-1), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        y_true.extend(labels.view(-1).cpu().numpy())
        y_pred.extend(torch.sigmoid(logits.view(-1)).detach().cpu().numpy())  # 使用detach()分离张量

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    # Compute evaluation metrics
    y_pred = np.array(y_pred)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    train_roc_auc = roc_auc_score(y_true, y_pred)
    train_roc_aucs.append(train_roc_auc)

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_loader):
            bg, labels = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            n_feats = bg.ndata.pop('h')
            e_feats = bg.edata.pop('e')
            n_feats = n_feats.to(device)
            e_feats = e_feats.to(device)
            labels = labels.to(torch.float32)
            logits = model(bg, n_feats, e_feats)
            test_loss += loss_fn(logits.view(-1), labels.view(-1)).item()
            y_true.extend(labels.view(-1).cpu().numpy())
            y_pred.extend(torch.sigmoid(logits.view(-1)).cpu().numpy())
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    # Compute evaluation metrics
    y_pred = np.array(y_pred)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    test_roc_auc = roc_auc_score(y_true, y_pred)
    if test_roc_auc > best_accuracy:
        best_accuracy = test_roc_auc
        best_model_state_dict = model.state_dict()
        torch.save(best_model_state_dict, "Weave_best_model.pt")
    test_roc_aucs.append(test_roc_auc)

    # Print evaluation metrics for this epoch
    print('Epoch [{}/{}], Train loss: {:.4f}, Test loss: {:.4f}, train_roc_auc: {:.4f}, test_roc_auc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, test_loss, train_roc_auc, test_roc_auc))

filename = 'Weave_single_task'
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # 写入标题行
    writer.writerow(["train_losses", "test_losses", "train_roc_aucs", "test_roc_aucs"])

    # 逐行写入数据
    for i in range(len(train_losses)):
        writer.writerow([train_losses[i], test_losses[i], train_roc_aucs[i], test_roc_aucs[i]])

# Plot the loss curve and the ROC curve for the final epoch
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_roc_aucs, label='Train ROC AUC')
plt.plot(test_roc_aucs, label='Test ROC AUC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()