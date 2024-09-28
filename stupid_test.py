import os
import pickle
from collections import Counter

import cv2

from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch

def read_specific_frame(video_path, frame_numbers):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count_frame = 0

    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        res, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        flip_frame = cv2.flip(frame, 1)
        frames.append(frame)
        frames.append(flip_frame)
    return frames

data = {}
data_train = []
with open('output.csv', 'r') as f:
    reader = f.readlines()
    for row in reader:
        data_spl = row.split(' ')
        name = data_spl[0].replace('.mp4', '')
        data[name] = [int(d) for d in data_spl[1:]]

with open('train_data_yappy/train.csv', 'r') as f:
    reader = f.readlines()
    reader = reader[1:]
    for row in reader:
        data_spl = row.split(',')
        data_train.append([data_spl[1], data_spl[4]])

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

bf = cv2.BFMatcher()
t = False

if True:
    test_name = 'a18324cf-b2ad-41e2-86b8-e6923c5fdc36'
    images = read_specific_frame(f'train_data_yappy/train_dataset/{test_name}.mp4', data[f'{test_name}'])
    inputs = processor(images, return_tensors="pt")
    outputs = model(**inputs)
    serch_files = data_train
    key_insert = {}
    for next_frame in range(len(images)):
        image_mask = outputs.mask[next_frame]
        image_indices = torch.nonzero(image_mask).squeeze()
        image_keypoints = outputs.keypoints[next_frame][image_indices]
        image_scores = outputs.scores[next_frame][image_indices]
        image_descriptors = outputs.descriptors[next_frame][image_indices].cpu().detach().numpy()

        if (len(serch_files)) > 0:
            test = []
            for name_dump in serch_files:
                if os.path.isfile(f'descr/{name_dump[0]}.pkl'):
                    with open(f'descr/{name_dump[0]}.pkl', "rb") as dump:
                        des = pickle.load(dump)
                else:
                    print(f'NOT EXIST descr/{name_dump[0]}.pkl')

                n = name_dump
                if n == f'{test_name}':
                    continue
                match_list = []
                matches = bf.knnMatch(image_descriptors, des['des'], k=2)
                similar_list = []
                try:

                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            similar_list.append([m])

                    match_list.append([len(similar_list) / len(image_descriptors)])

                    match_list = sorted(match_list, key=lambda l: l[0], reverse=True)
                    # print(name_dump, match_list[0][0])
                    if match_list[0][0] > 0:
                        print(name_dump, match_list[0][0])
                    if match_list[0][0] >= 0.2:
                        test.append([name_dump, match_list[0][0]])
                except:
                    pass
                    print(f"{name_dump}: ERROR")
            if len(test) > 0:
                test = sorted(test, key=lambda l: l[1], reverse=True)
                if not test[0][0] in key_insert:
                    key_insert[test[0][0]] = []
                key_insert[test[0][0]].append(test[0][1])
    print(key_insert)

# key_insert = Counter(key_insert)
# print(key_insert.most_common()[0])
# sorted(key_insert, reverse=True)

if False:
    for next_data in data_train:
        exist = False
        key_insert = {}
        key = next_data[0]
        value = data[key]
        print(key)
        images = read_specific_frame(f'train_data_yappy/train_dataset/{key}.mp4', value)
        inputs = processor(images, return_tensors="pt")
        outputs = model(**inputs)
        serch_files = os.listdir('descr')
        if os.path.exists(f"descr/{key}.pkl"):
            exist = True
        if not exist:
            dumpfile = open(f'descr/{key}.pkl', 'wb')
        for next_frame in range(len(images)):
            image_mask = outputs.mask[next_frame]
            image_indices = torch.nonzero(image_mask).squeeze()
            image_keypoints = outputs.keypoints[next_frame][image_indices]
            image_scores = outputs.scores[next_frame][image_indices]
            image_descriptors = outputs.descriptors[next_frame][image_indices].cpu().detach().numpy()

            if (len(serch_files)) > 0:
                test = []
                for name_dump in serch_files:
                    if key == name_dump.split('.')[0]:
                        continue
                    with open(f'descr/{name_dump}', "rb") as dump:
                        des = pickle.load(dump)

                    match_list = []
                    matches = bf.knnMatch(image_descriptors, des['des'], k=2)
                    similar_list = []
                    try:

                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                similar_list.append([m])

                            match_list.append([len(similar_list) / len(image_descriptors)])

                        match_list = sorted(match_list, key=lambda l: l[0], reverse=True)
                        # print(name_dump, match_list[0][0])
                        if match_list[0][0] >= 0.2:
                            test.append([name_dump, match_list[0][0]])
                    except:
                        pass

                if len(test) > 0:
                    test = sorted(test, key=lambda l: l[1], reverse=True)
                    if not test[0][0] in key_insert:
                        key_insert[test[0][0]] = []
                    key_insert[test[0][0]].append(test[0][1])
            if not exist:
                contents = {"id": next_frame, "des": image_descriptors}
                pickle.dump(contents, dumpfile)

        # if ()

        if not exist:
            dumpfile.close()
        if len(key_insert) > 0:
            max = 0
            k = None
            for q, value in key_insert.items():
                if len(value) > max:
                    k = q
                    max = len(value)

            if k is None:
                print(key, '->', 'None')
            else:
                with open(f"test_scr.csv", "a") as fr:
                    fr.write(f'{key},{q},{next_data[1]},{value}\n')
                print(key, '->', q)
                print(value)
        # if len(os.listdir(f'descr')) >= 2:
        #     break
            # break
        # print(key, value)
        # break