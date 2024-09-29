import os
import pickle
from collections import Counter

import joblib
import requests
import test1

import os

import cv2

from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch

from joblib import Parallel, delayed
import time

print(torch.cuda.is_available())
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

number_of_cpu = joblib.cpu_count()

data = {}
data_train = []
# with open('output.csv', 'r') as f:
#     reader = f.readlines()
#     for row in reader:
#         data_spl = row.split(' ')
#         name = data_spl[0].replace('.mp4', '')
#         data[name] = [int(d) for d in data_spl[1:]]

with open('test_yappy/test.csv', 'r') as f:
    reader = f.readlines()
    reader = reader[1:]
    for row in reader:
        data_spl = row.split(',')
        data_train.append([data_spl[1], data_spl[2], data_spl[0]])

# device = torch.device('cuda')

print("start")
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
print('process')
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
# model.to(device)
print('load')
t = False

if False:
    test_name = '5eb4127e-5694-492b-963c-6688522e9ad2'
    images = read_specific_frame(f'train_data_yappy/train_dataset/{test_name}.mp4', data[f'{test_name}'])
    inputs = processor(images, return_tensors="pt")
    outputs = model(**inputs)
    # serch_files = data_train
    serch_files = [['3726bb2d-3323-41f8-8eb2-0d7cf095d62b', '']]
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
                    break

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
                        pass
                        # print(name_dump, match_list[0][0])
                    if match_list[0][0] >= 0.2:
                        test.append([name_dump, match_list[0][0]])
                except:
                    pass
                    print(f"{name_dump}: ERROR")

                    # break
                    # pass
            if len(test) > 0:
                test = sorted(test, key=lambda l: l[1], reverse=True)
                if not test[0][0][0] in key_insert:
                    key_insert[test[0][0][0]] = []
                key_insert[test[0][0][0]].append(test[0][0][1])
    print(key_insert)

# key_insert = Counter(key_insert)
# print(key_insert.most_common()[0])
# sorted(key_insert, reverse=True)

# data_train = [['0fd09c1b-e19e-4b6e-84e6-35f9d4fc6f72', ''], ['025ee26a-7391-4f60-878a-7fc1928a967b', '0fd09c1b-e19e-4b6e-84e6-35f9d4fc6f72']]
tp = 0
fn = 0
count = 0
if True:
    for next_data in data_train[:1600]:
        count += 1
        # url = next_data[2]
        # response = requests.get(url)
        file_path = f'test_yappy/test_dataset/{next_data[0]}.mp4'
        # if response.status_code == 200:
        #     with open(file_path, 'wb') as file:
        #         file.write(response.content)

        value = test1.extract_frame_metadata(file_path)
        # print(values)
        # break
        exist = False
        key_insert = {}
        key = next_data[0]
        # value = data[key]
        print(key)
        images = read_specific_frame(file_path, value)
        inputs = processor(images, return_tensors="pt")
        # inputs.to('cuda')
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
                # test = []
                def find_mp(name_dump):
                    # print(name_dump)
                    # print("test")
                    bf = cv2.BFMatcher()

                    test_mp = []
                # for name_dump in serch_files:
                    if key == name_dump.split('.')[0]:
                        return []
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
                        if match_list[0][0] >= 0.51:
                            test_mp.append([name_dump, match_list[0][0]])
                    except:
                        pass
                    return test_mp

                # test = Parallel(n_jobs=8)(delayed(find_mp)(i) for i in serch_files)
                delayed_funcs = [delayed(find_mp)(i) for i in serch_files]
                parallel_pool = Parallel(n_jobs=joblib.cpu_count())
                # parallel_pool = Parallel(n_jobs=1)
                test = parallel_pool(delayed_funcs)
                test = list(filter(None, test))
                if len(test) > 0:
                    new_test = []
                    for m in test:
                        new_test.append(m[0])
                    test = new_test
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
                # if '' == next_data[1]:
                #     tp += 1
                # else:
                #     fn += 1
                with open(f"test_scrN.csv", "a") as fr:
                    fr.write(f'{next_data[2]},{next_data[0]},{next_data[1]},false,\n')
                # print(key, '->', 'None')
            else:
                # if k.split(".")[0] == next_data[1]:
                #     tp += 1
                # else:
                #     fn += 1
                with open(f"test_scrN.csv", "a") as fr:
                    fr.write(f'{next_data[2]},{next_data[0]},{next_data[1]},true,{k.split(".")[0]}\n')
        else:
            # if '' == next_data[1]:
            #     tp += 1
            # else:
            #     fn += 1
            with open(f"test_scrN.csv", "a") as fr:
                fr.write(f'{next_data[2]},{next_data[0]},{next_data[1]},false,\n')


        # print(f'score = {tp / (tp + fn)}', f'count images : {count}')

                # print(key, '->', q)
                # print(value)
        # if len(os.listdir(f'descr')) >= 2:
        #     break
            # break
        # print(key, value)
        # break

# print(f'score = {tp/(tp+fn)}')