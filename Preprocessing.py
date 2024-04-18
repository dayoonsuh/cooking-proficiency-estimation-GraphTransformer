import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import os
import seaborn as sns
import torch
import warnings
from copy import deepcopy
from itertools import combinations
from tqdm import tqdm
from typing import *
import random
# from ..utils import *
from EgoExo4D_Dataset import EgoExo4d
import pickle as pkl
warnings.filterwarnings('ignore')

def seed_it_all(seed):
    ''' Attempt to be Reproducible '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ranges(nums:Iterable):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def load_pickle(filename: str):
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'rb') as handle:
        data = pkl.load(handle)
    return data

def save_pickle(filename: str, data: dict) -> None:
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
        

## !!!!!!!!!!Group by Actions!!!!!!!!!!!!!!!!!
from EgoExo4D_Dataset import EgoExo4d
class Preprocessing:

    def __init__(self, data_dir:str, normalization:bool=True):
        self.data_dir = data_dir #data/hands_landmarks 
        self.normalization = normalization
        self.egoexo4d = EgoExo4d()
        self.subject_id_to_time = self.egoexo4d.subject_id_to_time
        

    def get_filenames(self):
        return sorted([file for file in os.listdir(self.data_dir)])
    
    def get_subject_id(self, file:str):
        return file[:-9]

    def get_hand_landmarks(self, data:dict, verbose:bool=False):
        idx = 1
        cpt = 0
        landmarks = {}
        for key in data.keys():
            if len(data[key].keys()) > 0:
                landmarks[idx] = data[key]
                idx += 1
            else:
                cpt += 1
        if verbose: print(len(data.keys()), len(landmarks.keys()), cpt)
        assert len(data.keys()) == len(landmarks.keys()) + cpt
        return landmarks

    def get_right_hand_landmarks(self, data:dict):
        landmarks = {}
        for key in data.keys():
            try:
                right_hand_landmarks = data[key]['Left']
                landmarks[key] = right_hand_landmarks
            except:
                pass
        return landmarks

    def normalization_hand_landmarks(self, data:dict):
        out = {}
        for idx, key in enumerate(data):
            lands = data[key].reshape((21,3))
            if idx == 0:
                origin = lands[0]
            lands -= origin
            out[key] = lands.reshape(-1)
        return out

    def get_right_hand_landmarks_by_ex(self, data:dict, settings:list, fps = 30):

        landmarks_all_actions = {label: dict() for label in self.egoexo4d.label_idx.keys()}
        for action in settings.keys():
            for timestamp in settings[action]:
                start_time = timestamp[0]
                end_time = timestamp[1]
                start_frame = int(start_time*fps)
                end_frame = int(end_time*fps)
                for key in data.keys():
                    try:
                        right_hand_landmarks = data[key]['Left']
                        if start_frame<=key<=end_frame:
                            landmarks_all_actions[action][key]=right_hand_landmarks
                    except:
                        pass

        if self.normalization:
            for action in landmarks_all_actions.keys():
                landmarks_all_actions[action] = self.normalization_hand_landmarks(landmarks_all_actions[action])
           
        # for key, value in landmarks_all_actions.items():
        #     emptylist = []
        #     if len(value) == 0:             
        #         emptylist.append(key)
        #         print(key)
        #     for key in emptylist:
        #         # print(key)
        #         del landmarks_all_actions[key]

        return landmarks_all_actions

    def run(self):
        
        filenames = self.get_filenames()
        print(filenames)

        all_subject_data = {}
        
        for file in tqdm(filenames):
            data = load_pickle(os.path.join(self.data_dir, file))
            subject_id = self.get_subject_id(file)
            landmarks = self.get_hand_landmarks(data)
            print(subject_id)
            
            all_actions = self.get_right_hand_landmarks_by_ex(landmarks, self.subject_id_to_time[subject_id],30)
            all_subject_data[subject_id]=all_actions
            # landmarks_ex1, landmarks_ex2, landmarks_ex3 = self.get_right_hand_landmarks_by_ex(landmarks, subject_id_to_time[subject_id],30)
            
            landmarks_all = self.get_right_hand_landmarks(landmarks)
            all_subject_data[subject_id]['all'] = landmarks_all
            # all_exercises = {'ex1': landmarks_ex1, 'ex2': landmarks_ex2, 'ex3': landmarks_ex3, 'all': landmarks_all}
            # all_subject_data[subject_id] = all_exercises
       
        # self.subject_cleaning(all_subject_data)

        return all_subject_data
    
    
if __name__ == "__main__":
    data_dir = "data/hands_landmarks"
    prep = Preprocessing(data_dir)
    raw_data = prep.run()
    save_pickle("data/learning/raw_data_action_grouped.pkl", raw_data)
    print("Saved pickle data!")