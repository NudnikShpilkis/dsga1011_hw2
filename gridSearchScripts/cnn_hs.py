import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import datetime as dt

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
    
sns.set_style('darkgrid')
sns.set_context("notebook")

random.seed(134)

from funcs import readData, loadEmbeddings, hwDataset, hwCollateFn, gridSearchCNN

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
PARAMS = {'num_epochs':NUM_EPOCHS,
          'hidden_size':100,
          'weight_decay':0,
          'vocab_size':len(id2char)}

for i in np.arange(50, 450, 50):
    print("Hidden Size - {}".format(i))
    PARAMS['hidden_size'] = i
    train_acc, val_acc = gridSearchCNN(DATA, PARAMS)
    
    df = pd.concat([pd.DataFrame({'X':np.arange(NUM_EPOCHS), 'Y':train_acc, 'Acc':'Train'}), 
                    pd.DataFrame({'X':np.arange(NUM_EPOCHS), 'Y':val_acc, 'Acc':'Val'})], axis=0)
    
    plt.figure()
    pp = sns.lineplot(data=df, x = 'X', y = 'Y', hue='Acc', style="Acc", legend= "brief")
    pp.set_title('Hidden Size: {} | Accuracy: {}'.format(i, max(val_acc)))
    pp.set_ylabel("Accuracy")
    pp.set_xlabel("Epoch")
    pp.get_figure().savefig('figures/cnn_hs_{}.png'.format(i), bbox_inches='tight')
