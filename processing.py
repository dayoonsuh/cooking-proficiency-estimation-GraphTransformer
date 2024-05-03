import argparse
import numpy as np
import cv2
import os
import shutil
import re
from tqdm import tqdm
import pickle as pkl
from typing import *

import mediapipe as mp

from utils import *


def body_landmarks_extraction(subject_id, frames_dir, filenames:List[str], saved_img:str, saved_body_landmarks:str):
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:

        all_frame_landmarks = {}

        for frame_id, file in tqdm(enumerate(filenames), total=len(filenames)):

            assert frame_id+1 == int(re.findall('[0-9]+', file)[0])

            all_frame_landmarks[frame_id+1] = {}

            image = cv2.imread(os.path.join(frames_dir, file))

            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            coordinates_landmarks = []

            annotated_image = image.copy()

            for body_landmarks in results.pose_landmarks.landmark:

                coordinates_landmarks.append(body_landmarks.x)
                coordinates_landmarks.append(body_landmarks.y)
                coordinates_landmarks.append(body_landmarks.z)

            all_frame_landmarks[frame_id+1] = np.array(coordinates_landmarks)

            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            cv2.imwrite(os.path.join(saved_img, 'frame_' + str(frame_id) + '.jpg'), annotated_image)
        
    save_pickle(os.path.join(saved_body_landmarks, subject_id +'_data.pkl'), all_frame_landmarks)



def sort_frame_id(path):
    all_frames_unsorted = os.listdir(path)
    part1 = sorted([frame for frame in all_frames_unsorted if len(frame) == 11])
    part2 = sorted([frame for frame in all_frames_unsorted if len(frame) == 12])
    part3 = sorted([frame for frame in all_frames_unsorted if len(frame) == 13])
    part4 = sorted([frame for frame in all_frames_unsorted if len(frame) == 14])
    part5 = sorted([frame for frame in all_frames_unsorted if len(frame) == 15])

    all_frames = part1 + part2 + part3 + part4 + part5
    return all_frames


def frames_extraction(video_dir:str, filename:str, save_dir:str):
    print(os.path.join(video_dir, filename))
    if os.path.exists(os.path.join(video_dir, filename)):
        vidcap = cv2.VideoCapture(os.path.join(video_dir, filename))
        success, image = vidcap.read()
        count = 0
        while success:
            if count % 100 == 0:
                print("frame {} ".format(count))
            cv2.imwrite(os.path.join(save_dir, 'frame_' + str(count+1) + '.jpg'), image)          
            success, image = vidcap.read()
            count += 1
    else:
        raise print("File {} not found".format(os.path.join(video_dir, filename)))


def video_generation(frames_path:str, saved_path:str, filename:str):
    img_array = []
    all_frames = sort_frame_id(frames_path)
    print("video generation ...")
    print(os.path.join(saved_path, filename))
    for frame_id in tqdm(all_frames):
        img = cv2.imread(os.path.join(frames_path, frame_id))
        scale_percent = 25
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
        height, width, _ = img.shape
        size = (width,height)
        img_array.append(img)
    print(img.shape)
    out = cv2.VideoWriter(os.path.join(saved_path, filename), cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()


def hand_landmarks_detection(subject_id, frames_dir, filenames:List[str], saved_dir:str, save_hands_landmarks:str):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        all_frame_landmarks = {}

        #filenames = filenames[:50]

        for frame_id, file in tqdm(enumerate(filenames), total=len(filenames)):

            #print(frame_id+1, int(re.findall('[0-9]+', file)[0]))

            assert frame_id+1 == int(re.findall('[0-9]+', file)[0])

            all_frame_landmarks[frame_id+1] = {}

            image = cv2.imread(os.path.join(frames_dir, file))
            #print(image.shape)

            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            #print(results.multi_handedness)
            #nb_hands = len(results.multi_handedness)

            if not results.multi_hand_landmarks:
                continue

            #image_height, image_width, _ = image.shape
            
            annotated_image = image.copy()
            
            for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):

                coordinates_landmarks = []

                for i in range(len(hand_landmarks.landmark)):
                    coordinates_landmarks.append(hand_landmarks.landmark[i].x)
                    coordinates_landmarks.append(hand_landmarks.landmark[i].y)
                    coordinates_landmarks.append(hand_landmarks.landmark[i].z)

                label = list(handedness.classification)[0].label

                all_frame_landmarks[frame_id+1][label] = np.array(coordinates_landmarks)

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            cv2.imwrite(os.path.join(saved_dir, 'frame_' + str(frame_id) + '.jpg'), annotated_image)

    save_pickle(os.path.join(save_hands_landmarks, subject_id +'_data.pkl'), all_frame_landmarks)





def main():

    args = parser.parse_args()

    subjects_id = sorted(os.listdir('data/egoexo4d_data/takes'))

    for subject_id in subjects_id: # if feature already exists(extracted before downloading )
        filename = subject_id+"_data.pkl"
        
        if filename in os.listdir('data/hands_landmarks'):
            print('Feature file already exists!!')
            continue

        print("Subject ID : {}".format(subject_id))  
 
        # video frames extraction
        video_dir ="data/egoexo4d_data/takes"
        save_dir = os.path.join('data/frames', subject_id)
        make_dirs(save_dir)
        print(f"Created {save_dir}  directory!")

        
        if args.frame_extraction:
            print("Frame extracting..")
            if os.path.exists(os.path.join(video_dir, subject_id + '/frame_aligned_videos/aria01_214-1.mp4')):
                frames_extraction(video_dir, subject_id + '/frame_aligned_videos/aria01_214-1.mp4', save_dir)
            else:
                frames_extraction(video_dir, subject_id + '/frame_aligned_videos/aria04_214-1.mp4', save_dir)
            print("Frame extracted!")


        # hands landmarks extraction
        if args.hands_landmarks_extraction:
            #frames_dir = os.listdir(os.path.join('../data/frames', os.path.join(subject_type, subject_id)))
            hands_landmarks_dir = os.path.join('data/hands_landmarks_frames', subject_id)
            save_hands_landmarks = 'data/hands_landmarks'
            make_dirs(hands_landmarks_dir)
            # make_dirs(save_hands_landmarks)
            print(f"Created {hands_landmarks_dir}  directory!")
            # print(f"Created {save_hands_landmarks}  directory!")

            hand_landmarks_detection(subject_id, save_dir, sort_frame_id(save_dir), hands_landmarks_dir, 
                                     save_hands_landmarks)

            # saved_path = os.path.join("data/hands_landmarks_frames/videos", subject_id)
            # make_dirs(saved_path)

            # video_generation(hands_landmarks_dir, "data/hands_landmarks_frames/videos", subject_id + '.avi')

            shutil.rmtree(save_dir)
            print("Deleted '%s' directory successfully" % save_dir)

            shutil.rmtree(hands_landmarks_dir)
            print("Deleted '%s' directory successfully" % hands_landmarks_dir)
    
    print("All video features extracted!")
    print(f"There are {len(os.listdir('data/hands_landmarks'))} features in data/hands_landmarks")

'''
        if body_landmarks_extraction:

            #frames_dir = os.listdir(os.path.join('../data/frames', os.path.join(subject_type, subject_id)))
            body_landmarks_dir = os.path.join('../data/body_landmarks_frames', subject_id)
            saved_body_landmarks = '../data/body_landmarks'
            saved_img = os.path.join('../data/body_landmarks_frames', subject_id)
            make_dirs(body_landmarks_dir)

            body_landmarks_extraction(subject_id, save_dir, sort_frame_id(save_dir), saved_img, saved_body_landmarks)

            saved_path = os.path.join("../data/body_landmarks_frames/videos", subject_id)
            make_dirs(saved_path)

            video_generation(saved_img, saved_path, subject_id + '.avi')

            #shutil.rmtree(save_dir)
            #print("Deleted '%s' directory successfully" % save_dir)
            '''




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--frame_extraction', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--hands_landmarks_extraction', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--body_landmarks_extraction', default=False, action=argparse.BooleanOptionalAction)
   

    main()
    # hands_landmarks_dir = os.path.join('data/hands_landmarks_frames/upenn_0714_Cooking_3_2')
    # save_hands_landmarks = 'data/hands_landmarks'
    # make_dirs(hands_landmarks_dir)
    # hand_landmarks_detection("upenn_0714_Cooking_3_2",'data/frames/upenn_0714_Cooking_3_2', sort_frame_id('data/frames/upenn_0714_Cooking_3_2'), hands_landmarks_dir, save_hands_landmarks)