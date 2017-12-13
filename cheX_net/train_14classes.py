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
log_dir = 'tensorboard/class_14/'
weights_dir = 'weights_tmpdir/'
weights_path = 'weights_dir/chexnet_init_weights_class_14_tf.h5'
chex_net = CheXNet(reduction=0.5, classes=14)
chex_net.load_weights(weights_path)
# optimizer
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# loss function
def unweighted_binary_crossentropy(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred))


chex_net.compile(optimizer=adam, loss=unweighted_binary_crossentropy)

# callbacks
model_checkpoint_valloss = ModelCheckpoint(weights_dir + 'chexnet_14_weights_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.hdf5', monitor='val_loss', save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
tensor_board = TensorBoard(log_dir=log_dir)

callbacks = [model_checkpoint_valloss, reduce_lr, tensor_board]

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
        y = np.empty((self.batch_size, 14))

        for i, npy in enumerate(npys_temp):
            X[i,:,:,:] = np.load(path + 'processed_npy2/' + npy)
            y[i,:] = labels[npy]

        return X, y

# fit CheXNet

batch_size = 16
epochs = 200

labels = {}
partition = {}
f = open(path + 'labels14.pickle', 'rb')
labels = pickle.load(f)
f.close()
f = open(path + 'partition14.pickle', 'rb')
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
