#! /usr/bin/python3
# Author : Kevin Feghoul

import torch
import torch.nn as nn



class LSTM(nn.Module):

    def __init__(self, input_size:int, hidden_size:int, num_layers:int, output_size:int, bidirectional:str):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
                            
        
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      out, (h_out, c_out) = self.lstm(x)
      out = out[:, -1, :]
      out = self.fc(out)
      return out