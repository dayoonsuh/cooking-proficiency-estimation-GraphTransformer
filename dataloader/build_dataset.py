#! /usr/bin/python3
# Author : Kevin Feghoul

import numpy as np
import os
import pickle as pkl
import torch
import mediapipe as mp
from tqdm import tqdm
from typing import *

from torch.utils.data import Dataset
from torch_geometric.data import Data

import sys
from EgoExo4D_Dataset import EgoExo4d
np.set_printoptions(threshold=sys.maxsize)


def load_pickle(filename: str):
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'rb') as handle:
        data = pkl.load(handle)
    return data


def save_pickle(filename: str, data) -> None:
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


def make_dirs(path: str):
    '''Make directory if not exists.'''
    if not os.path.exists(path):
        os.makedirs(path)


class UnimodalDataset(Dataset):

    def __init__(self, feature, target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        return item,label


class MultiModalDataset2(Dataset):

    def __init__(self, feature_1, feature_2, target):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.target = target
    
    def __len__(self):
        return len(self.feature_1)
    
    def __getitem__(self,idx):
        item1 = self.feature_1[idx]
        item2 = self.feature_2[idx]
        label = self.target[idx]
        return item1, item2, label


class MultiModalDataset3(Dataset):

    def __init__(self, feature_1, feature_2, feature_3, target):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.feature_3 = feature_3
        self.target = target
    
    def __len__(self):
        return len(self.feature_1)
    
    def __getitem__(self,idx):
        item1 = self.feature_1[idx]
        item2 = self.feature_2[idx]
        item3 = self.feature_3[idx]
        label = self.target[idx]
        return item1, item2, item3, label


def data_loading(path_data_dir:str, filenames:str, every:int=1) -> dict:
    all_data = {}
    for file in tqdm(filenames):  
        if file[0] == 'I':
            subject_id = file[:9]
        else:
            subject_id = file[:5]
        all_data[subject_id] = {}
        data = load_pickle(os.path.join(path_data_dir, file))
        for idx, frame_id in enumerate(list(data.keys())):
            if idx % every == 0:
                try:
                    all_data[subject_id][frame_id] = data[frame_id]['Left']
                except:
                    pass
    print("data loading completed ...")
    return all_data


def sliding_window(data:np.array, window_size:int, overlap:float) -> np.array:
    overlap = int(window_size * (overlap / 100)) # overlap = 0
    out = np.array([data[i:i+window_size] for i in range(0, len(data), window_size-overlap) if len(data[i:i+window_size]) == window_size])
    # len(data) is number of extracted frames
    # each element(total = total frames in a video//600) in "out": 600 63 numbers of 
    return out


def make_window(data:dict, window_size:int, overlap:float): # overlap = 0
    L = []
    for key in data.keys():
        L.append(data[key]) # append video frame numbers
    tmp = np.array(L) # 1 frame has 63 features (3 coordinates * 21 joints)
    # print(tmp.shape) # (number of extracted frames, 63)
    out = sliding_window(tmp, window_size, overlap)
    if out.shape[1] != window_size:
        print(out)
    return out


def unison_shuffled_copies(a, b):
    print(len(a), len(b))
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def unison_shuffled_copies2(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def unison_shuffled_copies3(a, b, c, d):
    assert len(a) == len(b) == len(c) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]


def create_dataset(data:dict, ex_id:str, subjects:List[str], window_size:int=60, overlap:float=0.0):
    
    dataset = EgoExo4d()
    rootdir2proficiency = dataset.rootdir2proficiency
    # print(f"rootdir2prof length: {len(rootdir2proficiency)}")
    # print(f"subjects: {subjects}")
    
    labels = [] 
    idx = 0
    all_data = []
    for subject_id in subjects:   
        # label = 1 if subject_id[0] == 'C' else 0
        if rootdir2proficiency[subject_id].startswith("N"):
            label = 0
        elif rootdir2proficiency[subject_id].startswith("E"):
            label = 1
        else:
            label = 2
 
        if idx == 0: # if first video,
            try:
                win_data = make_window(data[subject_id][ex_id], window_size, overlap)
                all_data = win_data
                labels.extend([label] * win_data.shape[0])
                idx = 1
            except:
                pass
        else: # from second video
            try:  
                win_data = make_window(data[subject_id][ex_id], window_size, overlap)
                # print(win_data)
                all_data = np.vstack((all_data, win_data)) 
                labels.extend([label] * win_data.shape[0])
            except:
                pass  
       
    X, Y = unison_shuffled_copies(np.array(all_data), np.array(labels))
    X, Y = np.array(all_data), np.array(labels)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y).long()
    return X, Y


def create_multimodal_dataset(all_data:dict, ex_id:str, subjects:List[str], window_size:int=60, overlap:float=0.0):
    labels = [] 
    idx = 0
    nb_modalities = len(list(all_data.keys()))
    for subject_id in subjects:   
        label = 1 if subject_id[0] == 'C' else 0
        if idx == 0:
            try:
                win_data1 = make_window(all_data['m1'][subject_id][ex_id], window_size, overlap)
                win_data2 = make_window(all_data['m2'][subject_id][ex_id], window_size, overlap)
                all_data1, all_data2 = win_data1, win_data2
                if nb_modalities == 3:
                    win_data3 = make_window(all_data['m3'][subject_id][ex_id], window_size, overlap)
                    all_data3 = win_data3
                labels.extend([label] * win_data1.shape[0])
                idx = 1
            except:
                print(subject_id)
        else:
            try:
                win_data1 = make_window(all_data['m1'][subject_id][ex_id], window_size, overlap)
                win_data2 = make_window(all_data['m2'][subject_id][ex_id], window_size, overlap)
                all_data1, all_data2 = np.vstack((all_data1, win_data1)), np.vstack((all_data2, win_data2)) 
                if nb_modalities == 3:
                    win_data3 = make_window(all_data['m3'][subject_id][ex_id], window_size, overlap)
                    all_data3 = np.vstack((all_data3, win_data3))
                labels.extend([label] * win_data1.shape[0])
            except: 
                pass  
    
    if nb_modalities == 3:
        X1, X2, X3, Y = unison_shuffled_copies3(np.array(all_data1), np.array(all_data2), np.array(all_data3), np.array(labels))
        X1, X2, X3, Y = torch.Tensor(X1), torch.Tensor(X2), torch.Tensor(X3), torch.Tensor(Y).long()
        return X1, X2, X3, Y
    else:
        X1, X2, Y = unison_shuffled_copies2(np.array(all_data1), np.array(all_data2), np.array(labels))
        X1, X2, Y = torch.Tensor(X1), torch.Tensor(X2), torch.Tensor(Y).long()
        return X1, X2, Y

'''
def create_connections(ws:int=600):
    connections = mp.solutions.holistic.HAND_CONNECTIONS
    all_connections = []

    for idx in range(ws):
        for connection in connections:

            x = connection[0] + (21 * idx)
            y = connection[1] + (21 * idx)
            all_connections.append((x, y))
            all_connections.append((y, x))

    for idx in range(21):
        all_id = [x for x in range(idx, ws, 21)]
        for i in range(len(all_id)-1):
            all_connections.append((all_id[i], all_id[i+1]))
            all_connections.append((all_id[i+1], all_id[i]))
    return all_connections


def create_graph(data:torch.Tensor, label: torch.Tensor, ws:int=600):
    ws = data.shape[0]
    x = data[0].reshape((21, 3))
    all_connections = create_connections(ws)
    for idx in range(1, ws):
        x = torch.cat((x, data[idx].reshape((21,3))), axis=0)
    edge_index = torch.tensor(all_connections, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.contiguous().T, y=label)
    return data
'''

def inter_finger_connections():
    ifc1 = [1,5,9,13,17]
    ifc2 = [2,6,10,14,18]
    ifc3 = [3,7,11,15,19]
    ifc4 = [4,8,12,16,20]
    all_if = [ifc1, ifc2, ifc3, ifc4]
    inter_finger_connection = []
    for l in all_if:
        for i in range(len(l)-1):
            inter_finger_connection.append((l[i],l[i+1]))
    return inter_finger_connection


def hand_joints_connections(ws:int, graph_type:str):
    
    mp_connections = mp.solutions.holistic.HAND_CONNECTIONS
    if_connections = inter_finger_connections()
    all_connections = []

    # spacial connections
    for idx in range(ws):
        for c in mp_connections:
            x = c[0] + (21 * idx)
            y = c[1] + (21 * idx)
            all_connections.append((x, y))
            all_connections.append((y, x))

        if graph_type == 'g2':
            for c in if_connections:
                x = c[0] + (21 * idx)
                y = c[1] + (21 * idx)
                all_connections.append((x, y))
                all_connections.append((y, x))

    # temporal connection
    for idx in range(21):
        all_id = [x for x in range(idx, ws*21, 21)]
        for i in range(len(all_id)-1):
            all_connections.append((all_id[i], all_id[i+1]))
            all_connections.append((all_id[i+1], all_id[i]))
    return all_connections


def create_graph(data:torch.Tensor, label:torch.Tensor, ws:int, graph_type:str):
    ws = data.shape[0]
    x = data[0].reshape((21, 3))
    all_connections = hand_joints_connections(ws, graph_type)
    for idx in range(1, ws):
        x = torch.cat((x, data[idx].reshape((21,3))), axis=0)
    edge_index = torch.tensor(all_connections, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.contiguous().T, y=label)
    return data


def hand_joints_connections_stgcn(graph_type:str):
    mp_connections = mp.solutions.holistic.HAND_CONNECTIONS
    if_connections = inter_finger_connections()
    all_connections = []
    for c in mp_connections:
        x, y = c[0], c[1]
        all_connections.append((x, y))
        all_connections.append((y, x))
    if graph_type == 'g2':
        for c in if_connections:
            x, y = c[0], c[1]
            all_connections.append((x, y))
            all_connections.append((y, x))


    out = np.array(all_connections).T
    adj = to_adj(out)

    return adj


def to_adj(data: np.array) -> np.array:
    num = data[0].max()
    adj = np.zeros((num+1, num+1))

    for idx in range(data.shape[1]):
        x, y = data[0][idx], data[1][idx]
        adj[x,y] = 1
        adj[y,x] = 1

    return adj

# if __name__ == '__main__':
    # data = load_pickle(os.path.join("data/learning", "raw_data.pkl"))
    # print(len(data))
    # print("LOADED DATA")
    # train_subjects = ['sfu_cooking_002_1', 'iiith_cooking_20_1', 'upenn_0710_Cooking_1_3']
    # train_subjects = ['sfu_cooking_002_1']
    # trainX, trainY = create_dataset(data, "all", train_subjects, 600, 0)
    # print(len(trainX))