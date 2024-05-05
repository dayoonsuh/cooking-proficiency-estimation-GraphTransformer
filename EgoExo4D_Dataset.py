"""The 21 hand landmarks."""
hand_landmarks_dict ={
  "WRIST" : 0,
  "THUMB_CMC" : 1,
  "THUMB_MCP" : 2,
  "THUMB_IP" : 3,
  "THUMB_TIP" : 4,
  "INDEX_FINGER_MCP" : 5,
  "INDEX_FINGER_PIP" : 6,
  "INDEX_FINGER_DIP" : 7,
  "INDEX_FINGER_TIP" : 8,
  "MIDDLE_FINGER_MCP" : 9,
  "MIDDLE_FINGER_PIP" : 10,
  "MIDDLE_FINGER_DIP" : 11,
  "MIDDLE_FINGER_TIP" : 12,
  "RING_FINGER_MCP" : 13,
  "RING_FINGER_PIP" : 14,
  "RING_FINGER_DIP" : 15,
  "RING_FINGER_TIP" : 16,
  "PINKY_MCP" : 17,
  "PINKY_PIP" : 18,
  "PINKY_DIP" : 19,
  "PINKY_TIP" : 20,
}
import pandas as pd
import numpy as np
import os
import json

class EgoExo4d():
    def __init__(self):
        video_ids_file_path = 'data/egoexo4d_data/takes'
        video_subsets_file_path = 'data/egoexo4d_data/ego4d_subset_ids.txt'
    
        RELEASE_DIR = "data/egoexo4d_data"  # NOTE: changeme

        self.egoexo = {
            "takes": os.path.join(RELEASE_DIR, "takes.json"),
            "demonstrator_train": os.path.join(RELEASE_DIR, "annotations", "proficiency_demonstrator_train.json"),
            "demonstrator_val": os.path.join(RELEASE_DIR, "annotations", "proficiency_demonstrator_val.json"),
            "keystep_train": os.path.join(RELEASE_DIR, "annotations", "keystep_train.json"),
            "keystep_val": os.path.join(RELEASE_DIR, "annotations", "keystep_val.json")       
        }
        
        for k, v in self.egoexo.items():
            self.egoexo[k] = json.load(open(v))
        keysteps = {**self.egoexo['keystep_train']['annotations'], **self.egoexo['keystep_val']['annotations']}
        
        self.video_uids = []
        self.video_names = self.get_video_names_from_txt(video_subsets_file_path)

        for take in self.egoexo["takes"]:
            if take['root_dir'][6:] in self.video_names:                
                self.video_uids.append(take['take_uid'])
         
        self.id2proficiency = self.get_id2proficiency(self.video_uids) #84
        self.id2rootdir = self.get_id2rootdir(self.video_uids) #84
        self.rootdir2proficiency = self.get_rootdir2proficiency(self.id2rootdir, self.id2proficiency) #84  
        self.label_idx = self.create_label_dict(self.video_uids, keysteps) # 37 verbs
        self.rootdir2time =self.create_rootdir2time(keysteps)
         
    def get_video_names_from_txt(self,filepath):
        with open(filepath, 'r') as f:
            videos = [video.strip() for video in f.readlines()]
        return videos
    
    def get_video_names(self, video_ids_file_path):
        video_names = os.listdir(video_ids_file_path)
        return video_names
    
    def create_label_dict(self, video_uids, annotations):
        verbs = set()
        for uid in video_uids:
            segments =  annotations[uid]['segments']
            for segment in segments:
                verbs.add(segment['step_name'].split(" ")[0])
        verbs = sorted(verbs)
        indexed_verbs={}
        for id, verb in enumerate(verbs):
            indexed_verbs[verb] = id

        '''{'Add': 0, 'Adjust': 1, 'Break': 2, 'Check': 3, 'Close': 4, 'Cook': 5, 'Crack': 6, 'Cut': 7, 'Fill': 8, 
        'Flip': 9, 'Fold': 10, 'Gently': 11, 'Get': 12, 'Heat': 13, 'Leave': 14, 'Lift': 15, 'Mix': 16, 'Peel': 17, 
        'Place': 18, 'Pour': 19, 'Put': 20, 'Remove': 21, 'Simmer': 22, 'Steep': 23, 'Stir': 24, 'Taste': 25, 'Throw': 26, 
        'Tilt': 27, 'Toss': 28, 'Transfer': 29, 'Turn': 30,'Visually': 31, 'Wait': 32, 'Warm': 33, 'Wash': 34, 'Whisk': 35, 'Wipe': 36} '''
        
        return indexed_verbs        
                
    def get_id2proficiency(self,video_uids):
        id2proficiency = {}        
        for annotation in self.egoexo['demonstrator_val']['annotations']:
            if annotation['take_uid'] in video_uids:
                id2proficiency[annotation['take_uid']] = annotation['proficiency_score']

        for annotation in self.egoexo['demonstrator_train']['annotations']:
            if annotation['take_uid'] in video_uids:
                id2proficiency[annotation['take_uid']] = annotation['proficiency_score']

        return id2proficiency
        
    def get_id2rootdir(self, video_uids):
        id2rootdir = {}
        for take in self.egoexo["takes"]:
            if take['take_uid'] in video_uids:
                id2rootdir[take['take_uid']] = take['root_dir'][6:] # {id: root_dir}

        return id2rootdir
                
    def get_rootdir2proficiency(self, rootdir, proficiency):
        ids = rootdir.keys()  
        rootdir2proficiency = {rootdir[id]:proficiency[id] for id in ids}
        
        return rootdir2proficiency

    def create_rootdir2time(self, annotations):
        sitt = {}
        for video_id, rootdir in self.id2rootdir.items():
            sitt[rootdir]={}
            for label in self.label_idx:
                sitt[rootdir][label]=[]
        for video_id in self.video_uids:
            rootdir = annotations[video_id]['take_name']
            segments = annotations[video_id]['segments']
            for segment in segments:
                verb = segment['step_name'].split(" ")[0]
                sitt[rootdir][verb].append([segment['start_time'], segment['end_time']])
        
        return sitt
 
    def write_rootdir2proficiency(self, dict1, dict2):
        
        filename= f"rootdir2proficiency.txt"
        dict1 = dict(sorted(dict1.items(), key=lambda item: item[1]))
        ids = dict1.keys()
        with open(filename, 'w') as f:
            for id in ids:
                f.write(dict1[id]+" " + dict2[id]+'\n')
        f.close()
                       
    