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
from models import *
from executor.trainer import *
from results import *
from utils import *

import torch.nn as nn
from torch.utils.data import DataLoader

import warnings

warnings.filterwarnings("ignore")


def main():
    
    args = parser.parse_args()

    seed_it_all(args.seed)

     # create a folder for saving results and models
    filename = time.strftime("%Y%m%d-%H%M%S")

    #nb_subject_folder = 'all_subjects' if args.nb_subject == -1 else str(args.nb_subject) + '_subjects'

    print('\nTraining configs :')
    print('\nSubject Indepente CV | exercise : {} | model : {} | features : {} | window size : {} | overlap : {}\n'.format(args.ex_id, args.model_name, 
            args.data_type, args.window_size, args.overlap))

    out_path = os.path.join('output', 'SI-CV', args.ex_id, 'unimodal',args.data_type, args.model_name, "window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), filename)
    make_dirs(out_path)

    all_results_path = os.path.join('output', 'SI-CV', args.ex_id, 'unimodal', args.data_type, args.model_name, "window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), "results.txt")

    best_results_path = os.path.join('output', 'SI-CV', args.ex_id, 'unimodal', args.data_type, args.model_name, "window_size_" + str(args.window_size) 
                           + '_' + str(args.overlap), "sort_results.txt")

    # saving all the command line arguments in the path_results directory
    path_command_line_args = os.path.join(out_path, "commandline_args.txt")
    
    with open(path_command_line_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # data loading
    # if args.data_type == 'raw':
    #     data = load_pickle(os.path.join("../data/learning", "raw_data.pkl"))
    # elif args.data_type == 'one':
    #     data = load_pickle(os.path.join("../data/learning", "one_data.pkl"))
    # elif args.data_type == 'mean':
    #     data = load_pickle(os.path.join("../data/learning", "mean_data.pkl"))
    # elif args.data_type == 'dist':
    #     data = load_pickle(os.path.join("../data/learning", "dist_data.pkl"))
    # elif args.data_type == 'speed':
    #     data = load_pickle(os.path.join("../data/learning", "speed_data.pkl"))
    # elif args.data_type == 'angles':
    #     data = load_pickle(os.path.join("../data/learning", "angles_data.pkl"))
    # elif args.data_type == 'SoCJ':
    #     data = load_pickle(os.path.join("../data/learning", "SoCJ_data.pkl"))
    # elif args.data_type == 'SoCJ9':
    #     data = load_pickle(os.path.join("../data/learning", "SoCJ9_data.pkl"))
    # else:
    #     raise ValueError

    # subject_id = list(data.keys())
    # surgeon_id =[s_id for s_id in subject_id if s_id.startswith('Chir')]
    # intern_id = [s_id for s_id in subject_id if s_id.startswith('Chir') is not True][:12]    


    # surgeon_split = surgeon_partition(surgeon_id)
    # intern_split = intern_partition(intern_id, 4)
    # intern_split += intern_split
    # full_partition = surgeon_interns_partition(surgeon_split, intern_split)
    #-----------------------------------------------------------------------
    
    
    # data = load_pickle(os.path.join("data/learning", "raw_data.pkl")) # ungrouped
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
        #print(f"\nTrain {train_subjects}")
        #print(f"\nTest {test_subjects}\n")

        # create_dataset
        trainX, trainY = create_dataset(data, args.ex_id, train_subjects, args.window_size, args.overlap)

        if args.model_name == 'random':
            trainX = torch.rand((trainX.shape))

        testX, testY = create_dataset(data, args.ex_id, test_subjects, args.window_size, args.overlap)
        print("trainX shape : {}, trainY shape : {}, testX shape : {}, testY shape : {}".format(trainX.shape, trainY.shape, testX.shape, testY.shape))

        print(torch.bincount(trainY))
        print(torch.bincount(testY))

        train = UnimodalDataset(trainX, trainY)
        test = UnimodalDataset(testX, testY)
        
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)

        dataloaders = {"train" : train_loader, "test" : test_loader}
        
        out_dim = 3
        # out_dim = 2

        # models
        if args.model_name == 'cnn':
            model = CNN(trainX.shape[2], out_dim)
        elif args.model_name == 'tcn':
            model = TCNModel(trainX.shape[2], [32, 64], kernel_size=3)
        elif args.model_name == 'rnn':
            model = RNN(trainX.shape[2], args.hidden_dim, args.num_layers, out_dim)
        elif args.model_name == 'lstm':
            model = LSTM(trainX.shape[2], args.hidden_dim, args.num_layers, out_dim, args.bidirectional)
        elif args.model_name == 'gru':
            model = GRU(trainX.shape[2], args.hidden_dim, args.num_layers, out_dim, args.bidirectional)
        elif args.model_name == 'transformer':
            model = TransformerEncoder(trainX.shape[2], trainX.shape[1], args.max_len, args.d_model, args.n_heads, args.num_layers, args.dim_feedforward, out_dim, args.dropout_transformers)
        elif args.model_name == 'dgsta':
            model = DG_STA(args.window_size, out_dim, 0.0)
        elif args.model_name == 'deepgru':
            model = DeepGRU(trainX.shape[2], out_dim)
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
            weights = torch.tensor(class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(trainY.numpy()), y = trainY.numpy()), dtype=torch.float)
            #weights = torch.tensor([0.1,0.9])
            criterion = nn.CrossEntropyLoss(weights.to(args.device), reduction='mean')
        else:
            print("Not using class weight !\n")
            criterion = nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.975, patience=20, threshold=0.0001, threshold_mode='abs', verbose=True)
            
        # model training
        trained_model, history, preds, labels = train_model(model.to(args.device), criterion, optimizer, args.apply_scheduler, scheduler, dataloaders, args.device, 
                                            args.batch_size, args.num_epochs, args.early_stopping, args.patience, len(train), len(test), args.print_every)

        # saving the model 
        model_path = os.path.join(out_path, 'models')
        make_dirs(model_path)

        filename = str(idx+1) + '_' + args.model_name + "_weights.pth"
        torch.save(trained_model.state_dict(), os.path.join(model_path, filename))
        print(labels)
        print(preds)

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
    parser.add_argument('--ex_id', type=str, default='Get', choices=['ex1', 'ex2', 'ex3', 'all'])
    parser.add_argument('--data_type', type=str, default='raw', choices=['raw', 'dist', 'speed', 'angles', 'one', 'mean', 'SoCJ', 'SoCJ9'])
    parser.add_argument('--window_size', type=int, default=600)
    parser.add_argument('--overlap', type=int, default=0)

    # models params
    parser.add_argument('--model_name', type=str, default='transformer', choices=['cnn', 'tcn', 'rnn', 'lstm', 'gru', 'transformer', 'dgsta', 'deepgru'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--class_weight', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--apply_scheduler', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--early_stopping', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--qkv', type=int, default=8, help='dimension for query, key and value')
    parser.add_argument('--max_len', type=int, default=5000)   
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4) 
    parser.add_argument('--dim_feedforward', type=int, default=1024)
    parser.add_argument('--dropout_transformers', type=float, default=0.2)

    parser.add_argument('--print_every', type=int, default=1)


    main()