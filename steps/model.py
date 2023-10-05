from abc import ABC, abstractmethod
import logging
import time
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F



class Model(ABC):
    '''
    Abstract class for all models
    '''

    @abstractmethod
    def train(self):
        pass


class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['emb_size'])
        self.linear = nn.Sequential(
            nn.Linear(config['emb_size']*config['context_window'], config['d_model']),
            nn.Tanh(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        print("model params:", sum([m.numel() for m in self.parameters()]))

    def forward(self, idx, targets=None):
    
        x = self.embedding(idx)
      
        x = x.view(-1,self.config['emb_size']*self.config['context_window'])
        logits = self.linear(x)

        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            return logits, loss

        else:
            return logits



class MLP(Model):

    def __init__(self,config):
        self.config = config
        self.model = SimpleModel(self.config)

    def train(self,X,y):

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
        )
        
        losses = []
        for epoch in range(self.config['epochs']):
            optimizer.zero_grad()

            #Get mini-batch
            ix = torch.randint(0,X.shape[0],(128,))
            xs = X[ix]
            ys = y[ix]
            logits, loss = self.model(xs, ys)
            losses += [loss.item()]

            loss.backward()
            optimizer.step()

        
        return self.model,pd.DataFrame(losses).plot()
    