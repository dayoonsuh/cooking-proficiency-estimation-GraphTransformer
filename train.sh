#!/bin/bash

# CNN
#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50  --overlap {o}" -p ei ex2 -p dt raw -p m cnn -p l 0.001 0.0001 -p ws 600 -p bs 16 32 64 -p o 0 | simple_gpu_scheduler --gpus 0

# TCN
#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 30 --overlap {o}" -p ei ex2 -p dt raw -p m tcn -p l 0.001 -p ws 600 -p bs 128 -p o 90 | simple_gpu_scheduler --gpus 0

# Transformer
#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --d_model {d_m} --n_heads {n_h} --num_layers {n_l} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50 --overlap {o}" -p ei ex2 -p dt raw -p m transformer -p l 0.001 -p d_m 128 512 -p n_h 8 -p n_l 1 2 3 -p ws 600 -p bs 16 -p o 0 | simple_gpu_scheduler --gpus 0

# RNN
#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --hidden_dim {hd} --num_layers {nl} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50  --overlap {o} " -p ei ex2 -p dt raw -p m rnn -p l 0.001 -p hd 128 -p nl 1 -p bs 64 -p ws 600 -p o 0 | simple_gpu_scheduler --gpus 0

# GRU
#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --hidden_dim {hd} --num_layers {nl} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50  --overlap {o} " -p ei ex2 -p dt raw -p m gru -p l 0.01 0.001 0.0001 -p hd 128 256 512 1024 -p nl 1 2 -p bs 16 32 64 -p ws 600 -p o 0 | simple_gpu_scheduler --gpus 0

# DeepGRU

#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50  --overlap {o} " -p ei ex2 -p dt raw -p m deepgru -p l 0.0001 -p bs 16 -p ws 600 -p o 0 | simple_gpu_scheduler --gpus 0

# LSTM
#simple_hypersearch "python3 train.py --ex_id {ei} --data_type {dt} --model_name {m} --lr {l} --hidden_dim {hd} --num_layers {nl} --num_epochs 15 --batch_size {bs} --window_size {ws} --patience 50  --overlap {o} " -p ei ex2 -p dt raw -p m lstm -p l 0.001 -p hd 512 -p nl 1 -p bs 16 -p ws 600 -p o 0 | simple_gpu_scheduler --gpus 0

# GCN
#simple_hypersearch "python3 train_gnn.py --graph_type {gt} --model_name gnn --operator {op} --lr {l} --hidden_dim {hd} --num_layers {nl} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50 --overlap {o}" -p gt g2 -p op gcn -p l 0.001 -p hd 8 32 64 128 -p nl 1 2 -p ws 600 -p bs 16 64 128 -p o 0 | simple_gpu_scheduler --gpus 0

# GCN-Transformer
#simple_hypersearch "python3 train_gnn.py --graph_type {gt} --model_name gnn_transformer --operator {op} --lr {l} --hidden_dim {hd} --num_layers {nl} --num_layers_trans {nlt} --n_heads {n_h} --num_epochs 15 --batch_size {bs} --window_size {ws} --patience 50 --overlap {o}" -p gt g1 -p op gcn -p l 0.001 -p hd 8 -p nl 2 -p n_h 8 -p nlt 1 -p ws 600 -p bs 16 -p o 0 | simple_gpu_scheduler --gpus 0

# STGCN
#simple_hypersearch "python3 train_stgcn.py --model_name {m} --block {b} --graph_type {gt} --lr {l} --act_func {af} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50 --overlap {o}" -p m graph_conv -p b arch2 -p gt g1 -p l 0.001 -p af glu -p ws 600 -p bs 32 -p o 0 | simple_gpu_scheduler --gpus 0

# ASTGCN
#simple_hypersearch "python3 train_astgcn.py --lr {l} --nb_block {nb} --filter_dim {fd} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50 --overlap {o}" -p l 0.01 -p nb 1 -p fd 1 4 2 -p ws 600 -p bs 16 32 64 -p o 0 | simple_gpu_scheduler --gpus 0

# STGCN2
#simple_hypersearch "python3 train_stgcn2.py --lr {l} --inter_dim {id} --temporal_kernel_size {tks} --edge_importance_weighting {eiw} --num_epochs 150 --batch_size {bs} --window_size {ws} --patience 50 --overlap {o}" -p l 0.01 0.001 0.0001 -p id 4 8 16 32 64 128 256 -p tks 3 5 7 9 11 15 -p eiw 1 0 -p ws 600 -p bs 16 32 64 -p o 0 | simple_gpu_scheduler --gpus 0