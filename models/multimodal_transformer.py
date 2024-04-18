#! /usr/bin/python3
# Author : Kevin Feghoul

from typing import *
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer




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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
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

class MMT_e(nn.Module):

    def __init__(self, n_features:int, feats_dim:List[int], seq_length:int, d_model:int, n_head:int, num_layer:int, dim_feedforward:int,
                num_classes:int, max_len:int=5000, dropout:float=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(MMT_e, self).__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.act = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

        self.project_inp_f1 = nn.Linear(feats_dim[0], d_model)
        self.project_inp_f2 = nn.Linear(feats_dim[1], d_model)

        self.pos_enc_f1 = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)
        self.pos_enc_f2 = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if n_features == 3:
            self.project_inp_f3 = nn.Linear(feats_dim[2], d_model)
            self.pos_enc_f3 = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)
        
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, n_head, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layer)

        self.output_layer = nn.Linear(d_model * n_features * seq_length, self.num_classes)


    def forward(self, x:List[Tensor]):
        r"""
        Args:
            X: List[(batch_size, seq_length, feat_dim)]  list of inputs tensors (each input represent a modality)
        Returns:
            output: (batch_size, num_classes)
        """
        
        if self.n_features == 2:

            x1, x2 = x[0], x[1]

            inp_f1 = x1.permute(1, 0, 2)                             
            proj_inp_f1 = self.project_inp_f1(inp_f1) * math.sqrt(self.d_model)      
            pe_inp_f1 = self.pos_enc_f1(proj_inp_f1)   

            inp_f2 = x2.permute(1, 0, 2)                             
            proj_inp_f2 = self.project_inp_f2(inp_f2) * math.sqrt(self.d_model)      
            pe_inp_f2 = self.pos_enc_f2(proj_inp_f2)

            pe_inp = torch.cat((pe_inp_f1, pe_inp_f2), dim=0)

        else:
            x1, x2, x3 = x[0], x[1], x[2]
            
            inp_f1 = x1.permute(1, 0, 2)                             
            proj_inp_f1 = self.project_inp_f1(inp_f1) * math.sqrt(self.d_model)      
            pe_inp_f1 = self.pos_enc_f1(proj_inp_f1)   

            inp_f2 = x2.permute(1, 0, 2)                             
            proj_inp_f2 = self.project_inp_f2(inp_f2) * math.sqrt(self.d_model)      
            pe_inp_f2 = self.pos_enc_f2(proj_inp_f2)

            inp_f3 = x3.permute(1, 0, 2)                             
            proj_inp_f3 = self.project_inp_f3(inp_f3) * math.sqrt(self.d_model)      
            pe_inp_f3 = self.pos_enc_f3(proj_inp_f3)

            pe_inp = torch.cat((pe_inp_f1, pe_inp_f2, pe_inp_f3), dim=0)

        
        out = self.transformer_encoder(pe_inp)                    
        out = self.act(out)                                  
        out = out.permute(1, 0, 2)                           
        out = self.dropout(out)                              
        out = out.reshape(out.shape[0], -1) 

        out = self.output_layer(out)

        return out
        

class MMT_i(nn.Module):

    def __init__(self, n_features:int, feats_dim:List[int], seqs_length:List[int], d_models:List[int], n_heads:List[int], 
                num_layers:List[int], dim_feedforward:int, num_classes:int, max_len:int=5000, dropout:float=0.1, 
                pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(MMT_i, self).__init__()
        
        self.n_features = n_features
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.act = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = 3
        self.dilation = 1
        self.hidden_size = 512

        self.d_model_f1 = d_models[0]
        self.d_model_f2 = d_models[1]

        self.project_inp_f1 = nn.Linear(feats_dim[0], d_models[0])
        self.project_inp_f2 = nn.Linear(feats_dim[1], d_models[1])

        self.pos_enc_f1 = get_pos_encoder(pos_encoding)(d_models[0], dropout=dropout*(1.0 - freeze), max_len=max_len)
        self.pos_enc_f2 = get_pos_encoder(pos_encoding)(d_models[1], dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer_f1 = TransformerEncoderLayer(d_models[0], n_heads[0], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
            encoder_layer_f2 = TransformerEncoderLayer(d_models[1], n_heads[1], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer_f1 = TransformerBatchNormEncoderLayer(d_models[0], n_heads[0], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
            encoder_layer_f2 = TransformerBatchNormEncoderLayer(d_models[1], n_heads[1], dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder_f1 = nn.TransformerEncoder(encoder_layer_f1, num_layers[0])
        self.transformer_encoder_f2 = nn.TransformerEncoder(encoder_layer_f2, num_layers[1])

        #self.fc1 = nn.Linear(d_models[0] * seqs_length[0], 512)
        #self.fc2 = nn.Linear(d_models[1] * seqs_length[1], 512)


        if self.n_features == 3:

            self.d_model_f3 = d_models[2]
            self.project_inp_f3 = nn.Linear(feats_dim[2], d_models[2])
            self.pos_enc_f3 = get_pos_encoder(pos_encoding)(d_models[2], dropout=dropout*(1.0 - freeze), max_len=max_len)

            if norm == 'LayerNorm':
                encoder_layer_f3 = TransformerEncoderLayer(d_models[2], n_heads[2], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
            else:
                encoder_layer_f3 = TransformerBatchNormEncoderLayer(d_models[2], n_heads[2], dim_feedforward, dropout*(1.0 - freeze), activation=activation)

            self.transformer_encoder_f3 = nn.TransformerEncoder(encoder_layer_f3, num_layers[2])

            #self.fc3 = nn.Linear(d_models[2] * seqs_length[2], 512)

        #dim_concat = d_models[0] * seqs_length[0] + d_models[1] * seqs_length[1] + d_models[2] * seqs_length[2]

        #self.fc_concat2 = nn.Linear(512*2, dim_feedforward_concat)
        #self.fc_concat3 = nn.Linear(512*3, dim_feedforward_concat)

        self.output_layer = nn.Linear(3 * d_models[0] * seqs_length[0], self.num_classes)


    
    
    def pipeline(self, x:Tensor, d_model:int, projection, pos_enc, transformer_encoder):
        r"""
        Args:
            X: (batch_size, seq_length, feat_dim)  

        Returns:
            output: (batch_size, seq_length * d_model)
        """

        inp = x.permute(1, 0, 2)      
        proj_inp = projection(inp) * math.sqrt(d_model)     
        pe_inp = pos_enc(proj_inp)                           
        out = transformer_encoder(pe_inp)                    
        out = self.act(out)                                  
        out = out.permute(1, 0, 2)                           
        out = self.dropout(out)                              
        out = out.reshape(out.shape[0], -1)  
        #print(out.shape)  
        #out = fc(out)            

        return out


    def forward(self, x:List[Tensor]):
        r"""
        Args:
            X: List[(batch_size, seq_length, feat_dim)]  list of inputs tensors (each input represent a modality)
        Returns:
            output: (batch_size, num_classes)
        """

        if self.n_features == 2:
            x1, x2 = x[0], x[1]
            out_f1 = self.pipeline(x1, self.d_model_f1, self.project_inp_f1, self.pos_enc_f1, self.transformer_encoder_f1)
            out_f2 = self.pipeline(x2, self.d_model_f2, self.project_inp_f2, self.pos_enc_f2, self.transformer_encoder_f2)
            out = F.relu(torch.cat((out_f1, out_f2), dim=1))
            out = self.output_layer(out)

        else:
            x1, x2, x3 = x[0], x[1], x[2]
            out_f1 = self.pipeline(x1, self.d_model_f1, self.project_inp_f1, self.pos_enc_f1, self.transformer_encoder_f1)
            out_f2 = self.pipeline(x2, self.d_model_f2, self.project_inp_f2, self.pos_enc_f2, self.transformer_encoder_f2)
            out_f3 = self.pipeline(x3, self.d_model_f3, self.project_inp_f3, self.pos_enc_f3, self.transformer_encoder_f3)
            out = F.relu(torch.cat((out_f1, out_f2, out_f3), dim=1))
            out = self.output_layer(out)

        return out



class MMT_l(nn.Module):

    def __init__(self, n_features:int, feats_dim:List[int], seqs_length:List[int], d_models:List[int], n_heads:List[int], 
                num_layers:List[int], dim_feedforward:int, num_classes:int, max_len:int=5000, dropout:float=0.1, 
                pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(MMT_l, self).__init__()
        
        self.n_features = n_features
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.act = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

        # 2 and 3 modalities

        self.d_model_1f = d_models[0]
        self.d_model_2f = d_models[1]

        self.project_inp_1f = nn.Linear(feats_dim[0], d_models[0])
        self.project_inp_2f = nn.Linear(feats_dim[1], d_models[1])

        self.pos_enc_1f = get_pos_encoder(pos_encoding)(d_models[0], dropout=dropout*(1.0 - freeze), max_len=max_len)
        self.pos_enc_2f = get_pos_encoder(pos_encoding)(d_models[1], dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer_1f = TransformerEncoderLayer(d_models[0], n_heads[0], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
            encoder_layer_2f = TransformerEncoderLayer(d_models[1], n_heads[1], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer_1f = TransformerBatchNormEncoderLayer(d_models[0], n_heads[0], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
            encoder_layer_2f = TransformerBatchNormEncoderLayer(d_models[1], n_heads[1], dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder_1f = nn.TransformerEncoder(encoder_layer_1f, num_layers[0])
        self.transformer_encoder_2f = nn.TransformerEncoder(encoder_layer_2f, num_layers[1])

        self.output_layer_1f = nn.Linear(d_models[0] * seqs_length[0], self.num_classes)
        self.output_layer_2f = nn.Linear(d_models[1] * seqs_length[1], self.num_classes)
        
        # 3 modalities only

        if self.n_features == 3:

            self.d_model_3f = d_models[2]
            self.project_inp_3f = nn.Linear(feats_dim[2], d_models[2])
            self.pos_enc_3f = get_pos_encoder(pos_encoding)(d_models[2], dropout=dropout*(1.0 - freeze), max_len=max_len)

            if norm == 'LayerNorm':
                encoder_layer_3f = TransformerEncoderLayer(d_models[2], n_heads[2], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
            else:
                encoder_layer_3f = TransformerBatchNormEncoderLayer(d_models[2], n_heads[2], dim_feedforward, dropout*(1.0 - freeze), activation=activation)
           
            self.transformer_encoder_3f = nn.TransformerEncoder(encoder_layer_3f, num_layers[2])

            self.output_layer_3f = nn.Linear(d_models[2] * seqs_length[2], self.num_classes)

    
    def pipeline(self, x:Tensor, d_model:int, projection, pos_enc, transformer_encoder, output_layer):

        inp = x.permute(1, 0, 2)      
        proj_inp = projection(inp) * math.sqrt(d_model)     
        pe_inp = pos_enc(proj_inp)                           
        out = transformer_encoder(pe_inp)                    
        out = self.act(out)                                  
        out = out.permute(1, 0, 2)                           
        out = self.dropout(out)                              
        out = out.reshape(out.shape[0], -1)   
        out = output_layer(out)            

        return out


    def forward(self, x:List[Tensor]):
        r"""
        Args:
            X: List[(batch_size, seq_length, feat_dim)]  list of inputs tensors (each input represent a modality)
        Returns:
            output: (batch_size, num_classes)
        """

        if self.n_features == 2:
            x1, x2 = x[0], x[1]
            #print(x1.shape, x2.shape)
            out1 = self.pipeline(x1, self.d_model_1f, self.project_inp_1f, self.pos_enc_1f, self.transformer_encoder_1f, self.output_layer_1f)
            out2 = self.pipeline(x2, self.d_model_2f, self.project_inp_2f, self.pos_enc_2f, self.transformer_encoder_2f, self.output_layer_2f)
            out_concat = torch.stack([out1, out2])

        else:

            x1, x2, x3 = x[0], x[1], x[2]
            #print(x1.shape, x2.shape, x3.shape)
            out1 = self.pipeline(x1, self.d_model_1f, self.project_inp_1f, self.pos_enc_1f, self.transformer_encoder_1f, self.output_layer_1f)
            out2 = self.pipeline(x2, self.d_model_2f, self.project_inp_2f, self.pos_enc_2f, self.transformer_encoder_2f, self.output_layer_2f)
            out3 = self.pipeline(x3, self.d_model_3f, self.project_inp_3f, self.pos_enc_3f, self.transformer_encoder_3f, self.output_layer_3f)
            out_concat = torch.stack([out1, out2, out3])

        out_mean = torch.mean(out_concat, axis=0)

        return out_mean

        











        





