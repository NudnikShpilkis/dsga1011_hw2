from collections import Counter
import random

import pickle as pkl

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import spacy
import datetime as dt

random.seed(134)

############################################################################
############################################################################
############################################################################
############################################################################

MAX_VOCAB_SIZE = 50000
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def buildVocab(data):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    tokens = []
    max_x1, max_x2 = 0, 0
    
    label_tokens = []
    
    for doc in data:
        max_x1 = max(max_x1, len(doc[0]))
        max_x2 = max(max_x2, len(doc[1]))
        tokens.extend(doc[0] + doc[1])
        label_tokens.append(doc[2])
    token_counter = Counter(tokens)
    vocab, count = zip(*token_counter.most_common(MAX_VOCAB_SIZE))
    vocab = ['<pad>', "<unk>"] + list(vocab)
    id2token = vocab
    token2id = dict(zip(vocab, range(len(vocab)))) 
    
    label_token_counter = Counter(label_tokens)
    label_vocab, label_count = zip(*label_token_counter.most_common(MAX_VOCAB_SIZE))
    id2label = list(label_vocab)
    label2id = dict(zip(label_vocab, range(0,len(label_vocab)))) 
    
    return token2id, id2token, max_x1, max_x2, label2id, id2label

### Function that preprocessed dataset
def readData():
    train_data = pkl.load(open("hw2_data/snli_train.p", "rb"))
    val_data = pkl.load(open("hw2_data/snli_val.p", "rb"))
    train_data = [[doc[0].split(" "), doc[1].split(" "), doc[2]] for doc in train_data]
    val_data = [[doc[0].split(" "), doc[1].split(" "), doc[2]] for doc in val_data]
    char2id, id2char, max_X1, max_x2, label2id, id2label = buildVocab(train_data)
    return train_data, val_data, char2id, id2char, max_X1, max_x2, label2id, id2label

############################################################################
############################################################################
############################################################################
############################################################################

def loadEmbeddings(char2id):

    word_embeddings = pkl.load(open("hw2_data/word_embeddings.p", "rb"))
    matrix_len = char2id
    weights_list = []
    words_found = 0

    for i, word in enumerate(char2id):
        if word == '<pad>':
            weights_list.append(torch.Tensor([0 for i in range(300)]))
            weights_list[-1].requires_grad = False
            continue
        elif word == "<unk>":
            weights_list.append(torch.rand(300))
            weights_list[-1].requires_grad = True
            continue
        try: 
            weights_list.append(torch.Tensor(word_embeddings[word]))
            weights_list[-1].requires_grad = False
        except KeyError:
            weights_list.append(torch.rand(300,))
            weights_list[-1].requires_grad = False

    weights_tensor = torch.stack(weights_list)
    return weights_tensor

############################################################################
############################################################################
############################################################################
############################################################################

MAX_X1 = 82
MAX_X2 = 41

class hwDataset(Dataset):
    def __init__(self, data_tuple, char2id, label2id):
        """
        @param data_list: list of character
        @param target_list: list of targets

        """
        self.x1, self.x2, self.target_list = zip(*data_tuple)
        assert (len(self.x1) == len(self.target_list))
        assert (len(self.x2) == len(self.target_list))
        assert (len(self.x1) == len(self.x2))
        self.char2id = char2id
        self.label2id = label2id

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        x1_idx = [self.char2id[c] if c in self.char2id.keys() else UNK_IDX  for c in self.x1[key][:MAX_X1]]
        x2_idx = [self.char2id[c] if c in self.char2id.keys() else UNK_IDX  for c in self.x2[key][:MAX_X2]]
        label = [self.label2id[self.target_list[key]]]
        
        return [x1_idx, len(x1_idx), 
                x2_idx, len(x2_idx), 
                torch.Tensor(label)]
    
def hwCollateFn(batch):
    x1_list = []
    x2_list = []
    label_list = []

    for datum in batch:
        label_list.append(datum[4])
        
        padded_vec_x1 = np.pad(np.array(datum[0]),
                                pad_width=((0, MAX_X1 - datum[1])),
                                mode="constant", constant_values=0)
        x1_list.append(padded_vec_x1)
        
        padded_vec_x2 = np.pad(np.array(datum[2]),
                                pad_width=((0, MAX_X2 - datum[3])),
                                mode="constant", constant_values=0)
        x2_list.append(padded_vec_x2)
    
    label_list = np.array(label_list)
    x1_list = np.array(x1_list)
    x2_list = np.array(x2_list)
    
    return [torch.from_numpy(x1_list),
            torch.from_numpy(x2_list), 
            torch.LongTensor(label_list)]

############################################################################
############################################################################
############################################################################
############################################################################

class RNNEncoder(nn.Module):
    def __init__(self, DATA, PARAMS, num_layers, num_classes):
        super(RNNEncoder, self).__init__()
        
        num_epochs = PARAMS['num_epochs']
        hidden_size = PARAMS['hidden_size']
        weights_tensor = DATA['weights_tensor']

        self.num_layers, self.hidden_size = num_layers, 4*hidden_size
        
        num_emb, emb_size = weights_tensor.size()
        self.embedding = nn.Embedding(num_emb, emb_size).from_pretrained(weights_tensor)
        
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x1, x2):              
        x1_embed = self.embedding(x1).float()
        x2_embed = self.embedding(x2).float()
        
        x1_gru_out = self.gru(x1_embed)[1]
        x2_gru_out = self.gru(x2_embed)[1]
        
        x1_gru_out = torch.cat([x1_gru_out[0,:,:], x1_gru_out[-1,:,:]], dim=1)
        x2_gru_out = torch.cat([x2_gru_out[0,:,:], x2_gru_out[-1,:,:]], dim=1)
        
        outputs = torch.cat([x1_gru_out, x2_gru_out], 1)
        
        logits = self.linear(outputs)
        return logits

def testModel(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for x1, x2, labels in loader:
        outputs = F.softmax(model(x1, x2), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)

############################################################################
############################################################################
############################################################################
############################################################################

def gridSearchRNN(DATA, PARAMS):
    weights_tensor = DATA['weights_tensor']
    train_loader = DATA['train_loader']
    val_loader = DATA['val_loader']
    
    num_epochs = PARAMS['num_epochs']
    hidden_size = PARAMS['hidden_size']
    vocab_size = PARAMS['vocab_size']
    weight_decay = PARAMS['weight_decay']
    
    model = RNNEncoder(DATA, PARAMS, num_layers=1, num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay)

    train_acc= np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        for i, (x1, x2, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
        train_acc[epoch] = testModel(train_loader, model)
        val_acc[epoch] = testModel(val_loader, model)
                
    return train_acc, val_acc

############################################################################
############################################################################
############################################################################
############################################################################

class CNNEncoder(nn.Module):
    def __init__(self, DATA, PARAMS, num_layers, num_classes):

        super(CNNEncoder, self).__init__()
        
        kernel_size = PARAMS['kernel_size']
        num_epochs = PARAMS['num_epochs']
        hidden_size = PARAMS['hidden_size']
        weights_tensor = DATA['weights_tensor']
        
        self.num_layers, self.hidden_size = num_layers, 2*hidden_size
        
        num_emb, emb_size = weights_tensor.size()
        self.embedding = nn.Embedding(num_emb, emb_size).from_pretrained(weights_tensor)
        
        
        self.conv1 = nn.Conv1d(emb_size, hidden_size, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=1) 

        self.linear = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x1, x2):      
        x1_batch_size, x1_seq_len = x1.size()
        x2_batch_size, x2_seq_len = x2.size()

        x1_embed = self.embedding(x1).float()
        x2_embed = self.embedding(x2).float()
        
        #Don't have to switch back
        x1_hidden = self.conv1(x1_embed.transpose(1,2)).transpose(1,2)
        x1_hidden = F.relu(x1_hidden.contiguous().view(-1, x1_hidden.size(-1))).view(x1_batch_size, x1_seq_len, x1_hidden.size(-1))
        x1_hidden = self.conv2(x1_hidden.transpose(1,2)).transpose(1,2)
        x1_hidden = F.relu(x1_hidden.contiguous().view(-1, x1_hidden.size(-1))).view(x1_batch_size, x1_seq_len, x1_hidden.size(-1))
        x1_hidden = x1_hidden.max(dim=1, keepdim=False)[0].squeeze(dim=1)
        
        x2_hidden = self.conv1(x2_embed.transpose(1,2)).transpose(1,2)
        x2_hidden = F.relu(x2_hidden.contiguous().view(-1, x2_hidden.size(-1))).view(x2_batch_size, x2_seq_len, x2_hidden.size(-1))
        x2_hidden = self.conv2(x2_hidden.transpose(1,2)).transpose(1,2)
        x2_hidden = F.relu(x2_hidden.contiguous().view(-1, x2_hidden.size(-1))).view(x2_batch_size, x2_seq_len, x2_hidden.size(-1))
        x2_hidden = x2_hidden.max(dim=1, keepdim=False)[0].squeeze(dim=1)

        outputs = torch.cat([x1_hidden, x2_hidden], 1)
        
        logits = self.linear(outputs)
        return logits

############################################################################
############################################################################
############################################################################
############################################################################


def gridSearchCNN(DATA, PARAMS):
    weights_tensor = DATA['weights_tensor']
    train_loader = DATA['train_loader']
    val_loader = DATA['val_loader']
    
    num_epochs = PARAMS['num_epochs']
    weight_decay = PARAMS['weight_decay']
    
    model = CNNEncoder(DATA, PARAMS, num_layers=2, num_classes=3)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay)

    train_acc= np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        for i, (x1, x2, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Forward pass
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
        train_acc[epoch] = testModel(train_loader, model)
        val_acc[epoch] = testModel(val_loader, model)
                
    return train_acc, val_acc
