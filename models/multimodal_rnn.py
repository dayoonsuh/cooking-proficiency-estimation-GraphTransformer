#! /usr/bin/python3
# Author : Kevin Feghoul

from typing import *

import torch
import torch.nn as nn



class MultimodalLSTM(nn.Module):

    def __init__(self, feats_dim:List[int], hidden_dim:List[int], num_layers:List[int], out_dim:int, bidirectional:bool):
        super(MultimodalLSTM, self).__init__()
        
        self.feats_dim = feats_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        
        self.lstm1 = nn.LSTM(feats_dim[0], hidden_dim[0], num_layers[0], True, bidirectional)
        self.lstm2 = nn.LSTM(feats_dim[1], hidden_dim[1], num_layers[1], True, bidirectional)
        self.lstm3 = nn.LSTM(feats_dim[2], hidden_dim[2], num_layers[2], True, bidirectional)

        self.fc = nn.Linear(3 * hidden_dim, out_dim)

        '''
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        '''

    def forward(self, x):

        #print(x[0].shape, x[1].shape, x[2].shape)

        out1, (_, _) = self.lstm1(x[0])
        out1 = out1[:, -1, :]

        out2, (_, _) = self.lstm2(x[1])
        out2 = out2[:, -1, :]

        out3, (_, _) = self.lstm3(x[2])
        out3 = out3[:, -1, :]

        out = torch.concat((out1, out2, out3), axis=1)

        #print(out.shape, out1.shape, out2.shape, out3.shape)
        
        # concat
        out = self.fc(out)

        return out 



class MMGRU_e(nn.Module):

    def __init__(self, n_features:int, feats_dim:List[int], hidden_dim:int, num_layers:int, out_dim:int, bidirectional:bool):
        super(MMGRU_e, self).__init__()
        
        self.n_features = n_features
        self.feats_dim = feats_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bidirectional = bidirectional


        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 128)

        inter_dim = 128

        if n_features == 2:
            self.gru = nn.GRU(feats_dim[0]+feats_dim[1], hidden_dim, num_layers, True, bidirectional)

        elif n_features == 3:
            self.gru = nn.GRU(feats_dim[0]+feats_dim[1]+feats_dim[2], hidden_dim, num_layers, True, bidirectional)

        self.fc = nn.Linear(hidden_dim, out_dim)

        '''
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        '''

    def pipeline(self, x, layer_norm, gru_layer, fc):
        
        out, _ = gru_layer(x)
        out = out[:, -1, :]
        out = layer_norm(out)
        out = fc(out)

        return out


    def forward(self, x):

        if self.n_features == 2:
            x1, x2 = x[0], x[1]
            concat_inp = torch.cat((x1, x2), dim=2)

            out = self.pipeline(concat_inp, self.layer_norm, self.gru, self.fc)
            
        else:
            x1, x2, x3 = x[0], x[1], x[2]
            concat_inp = torch.cat((x1, x2, x3), dim=2)
            out = self.pipeline(concat_inp, self.layer_norm, self.gru, self.fc)

        return out 


    
class MMGRU_i(nn.Module):

    def __init__(self, n_features:int, feats_dim:List[int], hidden_dim:List[int], num_layers:List[int], out_dim:int, bidirectional:bool):
        super(MMGRU_i, self).__init__()
        
        self.n_features = n_features
        self.feats_dim = feats_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bidirectional = bidirectional

        inter_dim = 128

        self.layer_norm1 = nn.LayerNorm(hidden_dim[0])
        self.layer_norm2 = nn.LayerNorm(hidden_dim[1])
        
        self.gru1 = nn.GRU(feats_dim[0], hidden_dim[0], num_layers[0], True, bidirectional)
        self.gru2 = nn.GRU(feats_dim[1], hidden_dim[1], num_layers[1], True, bidirectional)

        self.fc1 = nn.Linear(hidden_dim[1], inter_dim)
        self.fc2 = nn.Linear(hidden_dim[1], inter_dim)

        if n_features == 3:
            self.layer_norm3 = nn.LayerNorm(hidden_dim[2])
            self.gru3 = nn.GRU(feats_dim[2], hidden_dim[2], num_layers[2], True, bidirectional)
            self.fc3 = nn.Linear(hidden_dim[2], inter_dim)

        self.fc = nn.Linear(n_features * inter_dim, out_dim)

        '''
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        '''

    def pipeline(self, x, layer_norm, gru_layer, fc):
        
        out, _ = gru_layer(x)
        out = out[:, -1, :]
        out = layer_norm(out)
        out = fc(out)

        return out


    def forward(self, x):

        if self.n_features == 2:
            out1 = self.pipeline(x[0], self.layer_norm1, self.gru1, self.fc1)
            out2 = self.pipeline(x[1], self.layer_norm2, self.gru2, self.fc2)
            out = torch.concat((out1, out2), axis=1)

        else:
            out1 = self.pipeline(x[0], self.layer_norm1, self.gru1, self.fc1)
            out2 = self.pipeline(x[1], self.layer_norm2, self.gru2, self.fc2)
            out3 = self.pipeline(x[2], self.layer_norm3, self.gru3, self.fc3)
            out = torch.concat((out1, out2, out3), axis=1)

        out = self.fc(out)

        return out 


class MMGRU_l(nn.Module):

    def __init__(self, n_features:int, feats_dim:List[int], hidden_dim:List[int], num_layers:List[int], out_dim:int, bidirectional:bool):
        super(MMGRU_l, self).__init__()
        
        self.n_features = n_features
        self.feats_dim = feats_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bidirectional = bidirectional

        inter_dim = 128

        self.layer_norm1 = nn.LayerNorm(hidden_dim[0])
        self.layer_norm2 = nn.LayerNorm(hidden_dim[1])
        
        self.gru1 = nn.GRU(feats_dim[0], hidden_dim[0], num_layers[0], True, bidirectional)
        self.gru2 = nn.GRU(feats_dim[1], hidden_dim[1], num_layers[1], True, bidirectional)

        self.fc1 = nn.Linear(hidden_dim[1], out_dim)
        self.fc2 = nn.Linear(hidden_dim[1], out_dim)

        if n_features == 3:
            self.layer_norm3 = nn.LayerNorm(hidden_dim[2])
            self.gru3 = nn.GRU(feats_dim[2], hidden_dim[2], num_layers[2], True, bidirectional)
            self.fc3 = nn.Linear(hidden_dim[2], out_dim)


        '''
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        '''

    def pipeline(self, x, layer_norm, gru_layer, fc):
        
        out, _ = gru_layer(x)
        out = out[:, -1, :]
        out = layer_norm(out)
        out = fc(out)

        return out


    def forward(self, x):

        if self.n_features == 2:
            out1 = self.pipeline(x[0], self.layer_norm1, self.gru1, self.fc1)
            out2 = self.pipeline(x[1], self.layer_norm2, self.gru2, self.fc2)
            out_concat = torch.stack((out1, out2), axis=1)

        else:
            out1 = self.pipeline(x[0], self.layer_norm1, self.gru1, self.fc1)
            out2 = self.pipeline(x[1], self.layer_norm2, self.gru2, self.fc2)
            out3 = self.pipeline(x[2], self.layer_norm3, self.gru3, self.fc3)
            out_concat = torch.stack((out1, out2, out3), axis=1)

        out_mean = torch.mean(out_concat, axis=1)

        #print(out1.shape, out2.shape, out_concat.shape, out_mean.shape)

        return out_mean
