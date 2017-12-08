from skimage import io, transform
from cheX_net import CheXNet
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


path = '/home/qwerty/data/NIH/'
log_dir = '/tmp/tensor_board_logs/'
weights_dir = 'weights_dir/'
weights_path = 'chexnet_init_weights_tf.h5'
chex_net = CheXNet(reduction=0.5)
chex_net.load_weights(weights_path)
# optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# loss function
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    weights = 81.69844179651696 # neg / pos 81.69844179651696
    return K.mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=weights), axis=-1)

metrics = ['accuracy']

chex_net.compile(optimizer=adam, loss=weighted_binary_crossentropy, metrics=metrics)

# callbacks
model_checkpoint_valloss = ModelCheckpoint(weights_dir + 'chexnet_weights_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.hdf5', monitor='val_loss')
model_checkpoint_acc = ModelCheckpoint(weights_dir + 'chexnet_weights_epoch_{epoch:03d}_val_acc_{val_acc:.4f}.hdf5', monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
tensor_board = TensorBoard(log_dir=log_dir)

callbacks = [model_checkpoint_valloss, model_checkpoint_acc, reduce_lr, tensor_board]

# generator for train and validation data
class Data_generator(object):
    def __init__(self, shape=(224,224,3), batch_size=16):
        self.dim_x = shape[0]
        self.dim_y = shape[1]
        self.dim_z = shape[2]
        self.batch_size = batch_size

    def generate(self, labels, npys):
        while 1:
            imax = len(npys) // self.batch_size
            for i in range(imax):
                npys_temp = npys[i*self.batch_size: (i+1)*self.batch_size]
                X, y = self._generate(labels, npys_temp)
                yield X, y

    def _generate(self, labels, npys_temp):
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty(self.batch_size)

        for i, npy in enumerate(npys_temp):
            X[i,:,:,:] = np.load(path + 'processed_npy2/' + npy)
            y[i] = labels[npy]

        return X, y

# fit CheXNet

batch_size = 16
epochs = 200

labels = {}
partition = {}
f = open(path + 'labels2.pickle', 'rb')
labels = pickle.load(f)
f.close()
f = open(path + 'partition2.pickle', 'rb')
partition = pickle.load(f)
f.close()

training_generator = Data_generator(shape=(224,224,3), batch_size=batch_size).generate(labels, partition['train'])
validation_generator = Data_generator(shape=(224,224,3), batch_size=batch_size).generate(labels, partition['valid'])

chex_net.fit_generator( generator=training_generator,
                        steps_per_epoch=len(partition['train'])//batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=len(partition['valid'])//batch_size )
