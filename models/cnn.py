#! /usr/bin/python3
# Author : Kevin Feghoul

import torch.nn as nn



class CNN(nn.Module):
    
    def __init__(self, input_size:int, out_size:int):
        super(CNN, self).__init__()

        self.out_size = out_size

        self.conv_layer = nn.Sequential(
          nn.Conv1d(input_size, 32, 3),
          nn.BatchNorm1d(32),
          nn.ReLU(),
          #nn.Dropout(0.1),

          nn.Conv1d(32, 64, 3),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          #nn.Dropout(0.1),

          nn.Conv1d(64, 128, 3),
          nn.BatchNorm1d(128),
          nn.ReLU(),

          nn.Conv1d(128, 256, 3),
          nn.BatchNorm1d(256),
          nn.ReLU(),

          nn.MaxPool1d(2),
          nn.Flatten()
        )

        # 600 : 75776
        # 150 : 18176

        self.linear_layer = nn.Sequential(
            nn.Linear(75776, 128),
            nn.Linear(128, out_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_size, seq_length, feat_dim)  -->  (batch_size, feat_dim, seq_length)
        x = self.conv_layer(x)
        out = self.linear_layer(x)
        return out
