from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import cv2
import pickle
from sklearn import metrics
import numpy as np
from cheX_net import CheXNet

model_path = 'weights_tmpdir/chexnet_14_weights_epoch_030_val_loss_37.4567.hdf5'
path = '/home/qwerty/data/NIH/'

with open(path + 'labels14.pickle', 'rb') as f:
    labels = pickle.load(f)

with open(path + 'partition14.pickle', 'rb') as f:
    partition = pickle.load(f)
# partition['test'] = partition['test'][:100]
chex_net = CheXNet(reduction=0.5, classes=14)
chex_net.load_weights(model_path)

X_test = np.empty((len(partition['test']), 224, 224, 3), dtype=np.float32)
y_test = np.empty((len(partition['test']), 14), dtype=np.float32)

for i, npy in enumerate(partition['test']):
    X_test[i,:,:,:] = np.load(path + 'processed_npy2/' + npy)
    y_test[i,:] = labels[npy]

y_pred = chex_net.predict(X_test)
d_list = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

for d in range(14):
    try:
        print('{0:<20}\t{1:<1.6f}'.format(d_list[d], metrics.roc_auc_score(y_test[:,d], y_pred[:,d])))
    except:
        pass
'''
Atelectasis         	0.740757       0.8094
Cardiomegaly        	0.839839       0.9248
Effusion            	0.848640       0.8638
Infiltration        	0.677323       0.7345
Mass                	0.710978       0.8676
Nodule              	0.634380       0.7802
Pneumonia           	0.690357       0.7680
Pneumothorax        	0.712591       0.8887
Consolidation       	0.753521       0.7901
Edema               	0.862136       0.8878
Emphysema           	0.729699       0.9371
Fibrosis            	0.725272       0.8047
Pleural_Thickening  	0.699083       0.8062
Hernia              	0.742579       0.9164
'''
