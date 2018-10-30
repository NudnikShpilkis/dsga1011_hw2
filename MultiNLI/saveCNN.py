import random
import pickle as pkl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

random.seed(134)

from funcs import readData, loadEmbeddings, hwDataset, hwCollateFn, CNNEncoder, testModel

train_data, val_data, char2id, id2char, MAX_X1, MAX_X2, label2id, id2label = readData()
weights_tensor = loadEmbeddings(char2id)

BATCH_SIZE = 200

train_dataset = hwDataset(train_data, char2id, label2id)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=hwCollateFn,
                                           shuffle=True)
val_dataset = hwDataset(val_data, char2id, label2id)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=hwCollateFn,
                                           shuffle=True)
                                           
NUM_EPOCHS = 10

DATA = {'weights_tensor':weights_tensor,
        'train_loader':train_loader,
        'val_loader':val_loader}

CNN_PARAMS = {'num_epochs':NUM_EPOCHS,
          'hidden_size':250,
          'weight_decay':0.0001,
          'vocab_size':len(id2char),
          'kernel_size':3}

def saveCNN(DATA, PARAMS):
    weights_tensor = DATA['weights_tensor']
    train_loader = DATA['train_loader']
    val_loader = DATA['val_loader']
    
    num_epochs = PARAMS['num_epochs']
    hidden_size = PARAMS['hidden_size']
    vocab_size = PARAMS['vocab_size']
    weight_decay = PARAMS['weight_decay']
    
    model =  CNNEncoder(DATA, PARAMS, num_layers=2, num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay)

    train_acc= np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        for i, (x1, x2, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), "cnn.pt")
    torch.save(criterion.state_dict(), "cnnCriterion.pt")
    torch.save(optimizer.state_dict(), "cnnOptimizer.pt")
    return "finished" 

#saveCNN(DATA, CNN_PARAMS)

train_data =  pkl.load(open("hw2_data/mnli_train.p", "rb"))
val_data =  pkl.load(open("hw2_data/mnli_val.p", "rb"))
weights_tensor = loadEmbeddings(char2id)

BATCH_SIZE = 200
NUM_EPOCHS = 5

def trainCNN(DATA, PARAMS):
    weights_tensor = DATA['weights_tensor']
    train_loader = DATA['train_loader']
    val_loader = DATA['val_loader']
    
    num_epochs = PARAMS['num_epochs']
    hidden_size = PARAMS['hidden_size']
    vocab_size = PARAMS['vocab_size']
    weight_decay = PARAMS['weight_decay']
    
    cnn_model = CNNEncoder(DATA, CNN_PARAMS, num_layers=2, num_classes=3)
    cnn_model.load_state_dict(torch.load('cnn.pt'))
    criterion = torch.nn.CrossEntropyLoss()
    criterion.load_state_dict(torch.load('cnnCriterion.pt'))
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=3e-4, weight_decay=weight_decay)
    optimizer.load_state_dict(torch.load('cnnOptimizer.pt'))

    train_acc= np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))
        for i, (x1, x2, labels) in enumerate(train_loader):
            cnn_model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = cnn_model(x1, x2)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
        train_acc[epoch] = testModel(train_loader, cnn_model)
        val_acc[epoch] = testModel(val_loader, cnn_model)

    return train_acc, val_acc
    
accuracies = []

for genre in ['fiction', 'telephone', 'slate', 'government', 'travel']:
    print("Genre: {}".format(genre))
    train_genre_data = [[v[0], v[1], v[2]] for v in train_data if v[3] == genre]
    val_genre_data = [[v[0], v[1], v[2]] for v in val_data if v[3] == genre]

    train_dataset = hwDataset(train_genre_data, char2id, label2id)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=hwCollateFn,
                                               shuffle=True)
    val_dataset = hwDataset(val_genre_data, char2id, label2id)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=BATCH_SIZE,
                                               collate_fn=hwCollateFn,
                                               shuffle=True)

    DATA = {'weights_tensor':weights_tensor,
            'train_loader':train_loader,
            'val_loader':val_loader}

    CNN_PARAMS = {'num_epochs':NUM_EPOCHS,
          'hidden_size':250,
          'weight_decay':0.0001,
          'vocab_size':len(id2char),
          'kernel_size':3}
              
    train_acc, val_acc = trainCNN(DATA, CNN_PARAMS)
    accuracies.append([genre, train_acc, val_acc])
    
with open('cnnGenre.p', 'wb') as f:
    pkl.dump(accuracies, f)
    