import cv2
import pandas as pd
import numpy as np
import random
import re
import sys
import pickle
from tqdm import tqdm

p_num = 30805  # 30805
pid = [i for i in range(1, p_num + 1)]
random.seed(0)
random.shuffle(pid)
train_pid = pid[:int(p_num * 0.8)]
valid_pid = pid[int(p_num * 0.8):int(p_num * 0.9)]
test_pid = pid[int(p_num * 0.9):]

path = "/home/qwerty/data/NIH/"
pneumonia = 'pneumonia'

df = pd.read_csv(path + 'Data_Entry_2017.csv')

def make_data(df, pids):
    npy_names = []
    labels = {}
    for pid in tqdm(pids):
        for index, row in df[df['Patient ID'] == pid].iterrows():
            img = cv2.resize(cv2.imread(path + 'images/' + row['Image Index']), (224, 224)).astype(np.float32)
            if img.shape != (224, 224, 3):
                print(row['Image Index'])
                break
            img[:,:,0] = (img[:,:,0] - 103.94) * 0.017
            img[:,:,1] = (img[:,:,1] - 116.78) * 0.017
            img[:,:,2] = (img[:,:,2] - 123.68) * 0.017

            # mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225]

            npy_name = row['Image Index'].replace('png', 'npy')
            np.save(path + 'processed_npy2/' + npy_name, img)
            npy_names.append(npy_name)
            if re.search(pneumonia, row['Finding Labels'], re.IGNORECASE):
                labels[npy_name] = 1
            else:
                labels[npy_name] = 0
    return npy_names, labels

train_npy, train_labels = make_data(df, train_pid)
valid_npy, valid_labels = make_data(df, valid_pid)
test_npy, test_labels = make_data(df, test_pid)

labels = {**train_labels, **valid_labels, **test_labels}
partition = {'train': train_npy, 'test': test_npy, 'valid': valid_npy}
print(partition)
print(labels)
with open(path + 'labels2.pickle', 'wb') as f:
    pickle.dump(labels, f)

with open(path + 'partition2.pickle', 'wb') as f:
    pickle.dump(partition, f)
