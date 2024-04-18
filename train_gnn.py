#! /usr/bin/python3
# Author : Kevin Feghoul

import argparse
import numpy as np
import os
import json
import time
from typing import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight

from dataloader.build_dataset import *
from models import GNN, GCNTransformer
from executor.trainer import *
from results import *
from utils import *

import torch.nn as nn
from torch_geometric.loader import DataLoader

import warnings
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from EgoExo4D_Dataset import EgoExo4d
warnings.filterwarnings("ignore")




def main():
    
    args = parser.parse_args()

    seed_it_all(args.seed)

     # create a folder for saving results and models
    filename = time.strftime("%Y%m%d-%H%M%S")

    if args.model_name == 'gnn':
        model_name = args.operator
    else:
        model_name = args.operator + '_' + 'transformer'

    print('\nTraining configs :')
    print('\nSubject Indepente CV | exercise : {} | model : {} | window size : {} | overlap : {}\n'.format(args.ex_id, model_name, args.window_size, args.overlap))
    
    out_path = os.path.join('output', 'SI-CV', args.ex_id, 'unimodal', args.graph_type, model_name + "/window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), filename)
    make_dirs(out_path)

    all_results_path = os.path.join('output', 'SI-CV', args.ex_id,  'unimodal', args.graph_type, "results.txt")
    best_results_path =os.path.join('output', 'SI-CV', args.ex_id,  'unimodal', args.graph_type, "sort_results.txt")

    # saving all the command line arguments in the path_results directory
    path_command_line_args = os.path.join(out_path, "commandline_args.txt")
    
    with open(path_command_line_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # data loading

    data = load_pickle(os.path.join("data/learning", "raw_data_action_grouped.pkl")) # grouped
    # data = load_pickle(os.path.join("data/learning", "raw_data.pkl")) # ungrouped
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

    # print(len(Novice_split),len(EarlyExpert_split),len(IntermediateExpert_split))# 5 5 5 - for action grouped
       
    all_acc, all_f1, all_prec, all_rec, all_cm = [], [], [], [], []
    
    # for idx, (train_subjects, test_subjects) in enumerate(kf.split(subject_id)):

    for idx, (train_subjects, test_subjects) in enumerate(full_partition):
        # train_subjects = [subject_id[i] for i in train_subjects]
        # test_subjects= [subject_id[i] for i in test_subjects]
        print()
        # print(f"{len(train_subjects)} Train subject: {train_subjects}")
        # # print()
        # print(f"{len(test_subjects)} Test subject: {test_subjects}")

        print("\n#-------------------------- Fold {} --------------------------#\n".format(idx+1))

        # create_dataset
        trainX, trainY = create_dataset(data, args.ex_id, train_subjects, args.window_size, args.overlap)

        testX, testY = create_dataset(data, args.ex_id, test_subjects, args.window_size, args.overlap)
        print("trainX shape : {}, trainY shape : {}, testX shape : {}, testY shape : {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))
        
        print("\nDistribution labels: ")
        print(torch.bincount(trainY))
        print(torch.bincount(testY))
        print('\n')

        data_list_train = [create_graph(trainX[idx], trainY[idx], args.window_size, args.graph_type) for idx in tqdm(range(trainX.shape[0]), total=trainX.shape[0])]
        data_list_test = [create_graph(testX[idx], testY[idx], args.window_size, args.graph_type) for idx in tqdm(range(testX.shape[0]), total=testX.shape[0])]
    
        train_loader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(data_list_test, batch_size=args.batch_size, shuffle=True)


        dataloaders = {"train" : train_loader, "test" : test_loader}
        
        input_dim = 3
        out_dim = 3
 
        if args.model_name == 'gnn':
            model = GNN(args.operator, input_dim, args.hidden_dim, args.num_layers, out_dim)

        elif args.model_name == 'gnn_transformer':
            model = GCNTransformer(args.operator, input_dim, args.hidden_dim, args.num_layers ,args.num_layers_trans, args.window_size, 
                                   args.d_model, args.n_heads, args.dim_feedforward, args.batch_size, out_dim)
        else:
            raise NameError

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
        trained_model, history, preds, labels = train_model_gnn(model.to(args.device), criterion, optimizer, args.apply_scheduler, scheduler, dataloaders, args.device, 
                                            args.num_epochs, args.patience, trainX.shape[0], testX.shape[0], args.print_every)

        # saving the model 
        if args.saved_model:
            model_path = os.path.join(out_path, 'models')
            make_dirs(model_path)
            filename = str(idx+1) + '_' + model_name + "_weights.pth"
            torch.save(trained_model.state_dict(), os.path.join(model_path, filename))

        print(labels)
        print(preds)

        #all_acc.append(round(max(history['val_acc']).item(), ndigits=2))
        all_acc.append(round(accuracy_score(labels, preds.cpu().numpy())* 100, ndigits=2))
        all_f1.append(round(f1_score(labels, preds.cpu().numpy(), average='weighted') * 100, ndigits=2)) 
        all_prec.append(round(precision_score(labels, preds.cpu().numpy(), average='weighted'), ndigits=2) * 100)
        all_rec.append(round(recall_score(labels, preds.cpu().numpy(), average='weighted'), ndigits=2) * 100)

        cm = save_results(history, idx + 1, labels, preds.cpu(), out_path)
        all_cm.append(cm)

        print(round(max(history['val_acc']).item(), ndigits=2), accuracy_score(preds.cpu(), labels) * 100)

    all_results = [all_acc, all_f1, all_prec, all_rec, all_cm]
    save_results_all_folds(all_results, args.model_name, out_path, all_results_path, best_results_path, args.window_size, args.overlap)
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # general params
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--graph_type', type=str, default='g1', choices=['g1', 'g2'])
    parser.add_argument('--ex_id', type=str, default='all', choices=['ex1', 'ex2', 'ex3', 'all', 'Get', 'Cut'])
    # parser.add_argument('--window_size', type=int, default=600)
    parser.add_argument('--window_size', type=int, default=600)
    parser.add_argument('--overlap', type=int, default=0)

    # models params
    parser.add_argument('--model_name', type=str, default='gnn_transformer', choices=['gnn', 'gnn_transformer'])
    parser.add_argument('--operator', type=str, default='gcn', choices=['gcn', 'cheb', 'graphconv', 'gat', 'transformer'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50) # default was 1
    parser.add_argument('--class_weight', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--apply_scheduler', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_stopping', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=100)
    # parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_layers_trans', type=int, default=2)
    parser.add_argument('--bidirectional', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--qkv', type=int, default=8, help='dimension for query, key and value')
    parser.add_argument('--max_len', type=int, default=5000)   
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4) 
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout_transformers', type=float, default=0.20)
    parser.add_argument('--saved_model', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--print_every', type=int, default=1)


    main()
