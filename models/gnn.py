#! /usr/bin/python3
# Author : Kevin Feghoul


import math
from typing import *

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool



class GNN(nn.Module):
    def __init__(self, operator:str, node_dim:int, hidden_dim:int, num_layers:int, out_dim:int):
        super(GNN, self).__init__()

        self.operator = operator
        self.num_layers = num_layers
        self.K = 3

        self.convs = nn.ModuleList()

        if operator == 'gcn':
            self.convs.append(GCNConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        elif operator == 'cheb':
            self.convs.append(ChebConv(node_dim, hidden_dim, K=self.K))
            for _ in range(1, num_layers):
                self.convs.append(ChebConv(hidden_dim, hidden_dim, K=self.K))

        elif operator == 'graphconv':
            self.convs.append(GraphConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(GraphConv(hidden_dim, hidden_dim))
        
        elif operator == 'gat':
            self.convs.append(GATConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(GATConv(hidden_dim, hidden_dim))

        elif operator == 'transformer':
            self.convs.append(TransformerConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim))

        self.lin = nn.Linear(hidden_dim, out_dim)


    def forward(self, x, edge_index, batch, bs):

        for layer_idx in range(self.num_layers):
          x = self.convs[layer_idx](x, edge_index)
          x = x.relu()

        x = global_mean_pool(x, batch) 

        out = self.lin(x)
        
        return out


'''
class GNN(nn.Module):

    def __init__(self, node_dim:int, hidden_dim:int, out_dim:int):
        super(GNN, self).__init__()
        
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim//4)
        self.lin = nn.Linear((hidden_dim//4), out_dim)


    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch) 

        #x = torch.cat((x, self.lin_concat(concat_out)), dim=1)

        out = self.lin(x)

        return out
'''


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x:Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.d_model = d_model
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x:Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", in_casual=None):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src
        


class GCNTransformer(nn.Module):
    def __init__(self, operator:str, node_dim:int, hidden_dim:int, num_layers:int, num_layers_trans:int, seq_length:int, d_model:int, 
                 n_heads:int, dim_feedforward:int, batch_size:int, out_dim:int, dropout=0.1, pos_encoding='fixed', activation='gelu', 
                 norm='BatchNorm', freeze:bool=False):

        super(GCNTransformer, self).__init__()

        self.operator = operator
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.nb_joint = 21
        self.K = 3

        self.convs = nn.ModuleList()

        if operator == 'gcn':
            self.convs.append(GCNConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

        elif operator == 'cheb':
            self.convs.append(ChebConv(node_dim, hidden_dim, K=self.K))
            for _ in range(1, num_layers):
                self.convs.append(ChebConv(hidden_dim, hidden_dim, K=self.K))

        elif operator == 'graphconv':
            self.convs.append(GraphConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(GraphConv(hidden_dim, hidden_dim))
        
        elif operator == 'gat':
            self.convs.append(GATConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(GATConv(hidden_dim, hidden_dim))

        elif operator == 'transformer':
            self.convs.append(TransformerConv(node_dim, hidden_dim))
            for _ in range(1, num_layers):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim))

        self.d_model = d_model
        self.max_len = 5000
        self.feat_dim = self.nb_joint * self.hidden_dim


        self.project_inp = nn.Linear(self.feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(self.feat_dim, dropout=dropout*(1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(self.feat_dim, n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.feat_dim, n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)


        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers_trans)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)
    
        #self.output_layer = nn.Linear(d_model * seq_length, out_dim)

        self.output_layer = nn.Linear(self.feat_dim * seq_length, out_dim)


    def forward(self, x, edge_index, batch, bs):

        # GNN
        for layer_idx in range(self.num_layers):
          x = self.convs[layer_idx](x, edge_index)
          x = x.relu()

        #print(x.shape)
        x = x.reshape((bs, self.seq_length, self.nb_joint * self.hidden_dim))

        # Transformer
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, feat_dim)  -->  (seq_length, batch_size, feat_dim) 
        #x = self.project_inp(x) * math.sqrt(self.d_model)  #  (seq_length, batch_size, feat_dim)  -->  (seq_length, batch_size, d_model)
        #x = self.pos_enc(x)  # add positional encoding (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)
        x = self.act(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        x = self.dropout1(x)

        # Output
        out = x.reshape(x.shape[0], -1)  # (batch_size, seq_length * d_model)
        out = self.output_layer(out)  # (batch_size, num_classes)

        return out

