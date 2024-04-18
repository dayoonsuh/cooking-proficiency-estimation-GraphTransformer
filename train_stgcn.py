#! /usr/bin/python3
# Author : Kevin Feghoul

import argparse
import numpy as np
import os
import json
import time
from typing import *
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import class_weight

from dataloader.build_dataset import *
from models import STGCNGraphConv, STGCNChebGraphConv
from executor.trainer import *
from results import *
from utils import *

import torch.nn as nn
from torch_geometric.loader import DataLoader

import warnings

warnings.filterwarnings("ignore")




def main():
    
    args = parser.parse_args()

    seed_it_all(args.seed)

     # create a folder for saving results and models
    filename = time.strftime("%Y%m%d-%H%M%S")

    model_name = 'stgcn'

    print('\nTraining configs :')
    print('\nSubject Indepente CV | exercise : {} | model : {} | window size : {} | overlap : {}\n'.format(args.ex_id, model_name, args.window_size, args.overlap))
    
    out_path = os.path.join('out', 'SI-CV', args.ex_id, 'unimodal', args.graph_type, model_name + "/window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), filename)
    make_dirs(out_path)

    all_results_path = os.path.join('out', 'SI-CV', args.ex_id,  'unimodal', args.graph_type, model_name + "/window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), "results.txt")

    best_results_path = os.path.join('out', 'SI-CV', args.ex_id,  'unimodal', args.graph_type, model_name + "/window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), "sort_results.txt")

    # saving all the command line arguments in the path_results directory
    path_command_line_args = os.path.join(out_path, "commandline_args.txt")
    
    with open(path_command_line_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # data loading
    # data = load_pickle(os.path.join("../data/learning", "raw_data.pkl"))

    # # SICV
    # subject_id = list(data.keys())
    # surgeon_id =[s_id for s_id in subject_id if s_id.startswith('Chir')]
    # intern_id = [s_id for s_id in subject_id if s_id.startswith('Chir') is not True][:12]    

    # surgeon_split = surgeon_partition(surgeon_id)
    # intern_split = intern_partition(intern_id, 4)
    # intern_split += intern_split
    # full_partition = surgeon_interns_partition(surgeon_split, intern_split)
    data = load_pickle(os.path.join("data/learning", "raw_data_action_grouped.pkl")) # grouped
    print("LOADED DATA")

    dataset = EgoExo4d()

    subject_id = dataset.rootdir2proficiency.keys() # action grouped
    
    rootdir2proficiency = dataset.rootdir2proficiency
    #Action Grouped
    Novice_id = [s_id for s_id in subject_id if rootdir2proficiency[s_id].startswith("Novice")] # 24
    EarlyExpert_id = [s_id for s_id in subject_id if rootdir2proficiency[s_id].startswith('Early')] #24
    IntermediateExpert_id = [s_id for s_id in subject_id if rootdir2proficiency[s_id].startswith('Intermediate')] #18
    #----------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------
    Novice_split = level_partition(Novice_id,8) # train:32, test:8
    EarlyExpert_split = level_partition(EarlyExpert_id,8) # train:32, test:8
    IntermediateExpert_split = level_partition(IntermediateExpert_id,8) # train:32, test:8
    full_partition = surgeon_interns_partition(Novice_split,EarlyExpert_split,IntermediateExpert_split)

    all_acc, all_f1, all_prec, all_rec, all_cm = [], [], [], [], []

    for idx, (train_subjects, test_subjects) in enumerate(full_partition):

        print("\n#-------------------------- Fold {} --------------------------#\n".format(idx+1))

        # create_dataset
        trainX, trainY = create_dataset(data, args.ex_id, train_subjects, args.window_size, args.overlap)

        testX, testY = create_dataset(data, args.ex_id, test_subjects, args.window_size, args.overlap)
        print("trainX shape : {}, trainY shape : {}, testX shape : {}, testY shape : {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))
        
        print("\nLabel distribution: ")
        print(torch.bincount(trainY))
        print(torch.bincount(testY))
        print('\n')

        edge_index = torch.Tensor(hand_joints_connections_stgcn(args.graph_type)).to(args.device)

        train = UnimodalDataset(trainX, trainY)
        test = UnimodalDataset(testX, testY)
        
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

        dataloaders = {"train" : train_loader, "test" : test_loader}
        
        input_dim = 3
        out_dim = 2


        configs = {
            'Kt': 3,
            'Ks': 3,
            'n_his': args.window_size,
            'n_vertex': 21,
            'act_func': args.act_func,
            'gso': edge_index,
            'enable_bias': True,
            'droprate': 0.0,
            'graph_conv_type': args.model_name,   # 'cheb_graph_conv', 'graph_conv'
            'ws': args.window_size
        }

        #blocks = [[3], [64, 16, 64], [64, 16, 64], [128, 128], [1]]

        n_joint = 21

        if args.block == 'arch1':
            blocks = [[3], [64, 16, 64], [128, 128], [1]]
        elif args.block == 'arch2':
            blocks = [[3], [64, 16, 64], [64, 16, 64], [128, 128], [1]]
        elif args.block == 'arch3':
            blocks = [[3], [32, 8, 32], [64, 64], [1]]
        elif args.block == 'arch4':
            blocks = [[3], [32, 8, 32], [32, 8, 32], [64, 64], [1]]
        elif args.block == 'arch5':
            blocks = [[3], [32, 8, 32], [32, 8, 32], [32, 8, 32], [64, 64], [1]]
        elif args.block == 'arch6':
            blocks = [[3], [64, 128, 64], [128, 128], [1]]
        elif args.block == 'arch7':
            blocks = [[3], [64, 128, 256], [512, 512], [1]]
        
        
        

        if args.model_name == 'graph_conv':
            model = STGCNGraphConv(configs, blocks, n_joint)
        else:
            model = STGCNChebGraphConv(configs, blocks, n_joint)


        # calculates the total number of parameters of the model
        total_params = 0
        for _, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print("\nTotal Trainable Parameters : {}\n".format(total_params))

        # loss function
        if args.class_weight:
            weights = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(trainY.numpy()), 
                                     y = trainY.numpy()), dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weights.to(args.device), reduction='mean')
        else:
            print("Not using class weight !\n")
            criterion = nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.975, patience=20, threshold=0.0001, threshold_mode='abs', verbose=True)
            
        
        # model training
        trained_model, history, preds, labels = train_model_stgcn(model.to(args.device), edge_index, criterion, optimizer, args.apply_scheduler, scheduler, dataloaders, 
                                                                  args.device,args.num_epochs, args.patience, trainX.shape[0], testX.shape[0], args.print_every)

        # saving the model 
        if args.saved_model:
            model_path = os.path.join(out_path, 'models')
            make_dirs(model_path)
            filename = str(idx+1) + '_' + model_name + "_weights.pth"
            torch.save(trained_model.state_dict(), os.path.join(model_path, filename))

        all_acc.append(round(max(history['val_acc']).item(), ndigits=2))
        #all_acc.append(accuracy_score(labels, preds.cpu().numpy()))
        all_f1.append(round(f1_score(preds.cpu().numpy(), labels, average='weighted') * 100, ndigits=2)) 
        all_prec.append(round(precision_score(labels, preds.cpu().numpy(), average='weighted'), ndigits=2) * 100)
        all_rec.append(round(recall_score(labels, preds.cpu().numpy(), average='weighted'), ndigits=2) * 100)

        cm = save_results(history, idx + 1, labels, preds.cpu(), out_path)
        all_cm.append(cm)

    all_results = [all_acc, all_f1, all_prec, all_rec, all_cm]
    save_results_all_folds(all_results, args.model_name, out_path, all_results_path, best_results_path, args.window_size, args.overlap)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--graph_type', type=str, default='g1', choices=['g1', 'g2'])
    parser.add_argument('--ex_id', type=str, default='Add', choices=['ex1', 'ex2', 'ex3', 'all'])
    parser.add_argument('--nb_subject', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=600)
    parser.add_argument('--overlap', type=int, default=0)

    # models params
    parser.add_argument('--model_name', type=str, default='gnn_transformer', choices=['graph_conv', 'cheb_graph_conv'])
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu', 'relu', 'leaky_relu', 'silu'])
    parser.add_argument('--block', type=str, default='arch1', choices=['arch1', 'arch2', 'arch3', 'arch4', 'arch5', 'arch6'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--class_weight', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--apply_scheduler', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_stopping', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--saved_model', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--print_every', type=int, default=1)


    main()
