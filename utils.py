#! /usr/bin/python3
# Author : Kevin Feghoul

import itertools
import numpy as np
import random
import seaborn as sns
import os
import pickle as pkl
import torch
from copy import deepcopy
from matplotlib import rc
from pylab import rcParams
from typing import Any, List, NoReturn, Tuple


sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8


def seed_it_all(seed: int):
    ''' Attempt to be Reproducible '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_pickle(filename: str):
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'rb') as handle:
        data = pkl.load(handle)
    return data


def save_pickle(filename: str, data: dict) -> None:
    '''Save the data in the pickle (pkl) format.'''
    with open(filename, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


def make_dirs(path: str):
    '''Make directory if not exists.'''
    if not os.path.exists(path):
        os.makedirs(path)


def append_new_line(filename :str, text_to_append) -> NoReturn:
    '''Append given text as a new line at the end of the file.'''
    # Open the file in append & read mode ('a+')
    with open(filename, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)



def shuffle_list(L: List[Any], seed: int = 42) -> List[Any]:
    L_copy = L.copy()
    random.Random(seed).shuffle(L_copy)
    return L_copy


def concat(L: List[Any]) -> List[Any]:
    out = []
    for l in L:
        out += l
    return out


def surgeon_partition(surgeons_id: List[str]) -> List[Tuple[List[str], List[str]]]:
    out = []
    for subset in itertools.combinations(surgeons_id, 2):
        test_id = list(subset)
        train_id = deepcopy(surgeons_id)
        for id in subset:
            train_id.remove(id)
        out.append((train_id, test_id))
    # print(f"SUrgeons: {out}")
    return out


def level_partition(subjects: List[str], n: int, seed: int = 7777) -> List[Tuple[List[str], List[str]]]:
    if len(subjects)<=20:
        subjects += subjects
    lack = 40-len(subjects)
    if lack > 0:
        for _ in range(lack):
            rand_index = random.randint(0,len(subjects)-1)
            
            subjects.append(subjects[rand_index])
            
    subjects = shuffle_list(subjects, seed)
            
    groups_subjects = [subjects[idx:idx+n] for idx in range(0, len(subjects), n)]
    combinations = []
    for idx, group_subjects in enumerate(groups_subjects):
        tmp = groups_subjects.copy()
        tmp.pop(idx)
        combinations.append((concat(tmp), concat([group_subjects])))
 
    # print(f"Combinations: {combinations}")
    return combinations


def surgeon_interns_partition(
    Novice_groups: List[Tuple[List[str], List[str]]], 
    EarlyExpert_groups: List[Tuple[List[str], List[str]]],
    IntermediateExpert_groups: List[Tuple[List[str], List[str]]]
    ) -> List[Tuple[List[str], List[str]]]:

    out = []
    for Novice_group, EarlyExpert_group, IntermediateExpert in zip(Novice_groups, EarlyExpert_groups,IntermediateExpert_groups):
        train_id_n, test_id_n = Novice_group
        train_id_ee, test_id_ee = EarlyExpert_group
        train_id_ie, test_id_ie = IntermediateExpert
        
        out.append((train_id_n + train_id_ee + train_id_ie, test_id_n + test_id_ee +test_id_ie))
    return out

## ORIGINAL function
# def surgeon_interns_partition(
#     surgeons_groups: List[Tuple[List[str], List[str]]], 
#     interns_groups: List[Tuple[List[str], List[str]]]
#     ) -> List[Tuple[List[str], List[str]]]:

#     out = []
#     for surgeons_group, interns_group in zip(surgeons_groups, interns_groups):
#         train_id_s, test_id_s = surgeons_group
#         train_id_i, test_id_i = interns_group
#         out.append((train_id_s + train_id_i, test_id_s + test_id_i))
#     return out

