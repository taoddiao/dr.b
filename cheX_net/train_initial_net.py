from skimage import io, transform
from cheX_net import CheXNet
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import pandas as pd
import numpy as np
from tqdm import tqdm


path = "/home/qwerty/data/NIH/"


chex_net = CheXNet(reduction=0.5)

# optimizer
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# loss function
def weighted_binary_crossentropy(y_true, y_pred):
    epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    weights = 81.86770140428678 # neg / pos
    return K.mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=weights), axis=-1)


chex_net.compile(optimizer=adam, loss=weighted_binary_crossentropy, metrics=['accuracy'])

# callbacks
model_checkpoint = ModelCheckpoint('chexnet_weights.hdf5', monitor='loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)
tensor_board = TensorBoard(log_dir='/tmp/tensor_board_logs')

callbacks = [model_checkpoint, reduce_lr, tensor_board]

class Data_generator(object):
    def __init__(self, shape=(224,224), batch_size=16):
        self.dim_x = shape[0]
        self.dim_y = shape[1]
        self.batch_size = batch_size

    def generate(self, labels, npys):
        while 1:
            imax = len(npys) // self.batch_size
            for i in range(imax):
                npys_temp = npys[i*self.batch_size: (i+1)*self.batch_size]
                X, y = self._generate(labels, npys_temp)
                yield X, y

    def _generate(self, labels, npys_temp):
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, 1))
        y = np.empty(self.batch_size)

        for i, npy in enumerate(npys_temp):
            X[i,:,:,0] = np.load(path + 'processed_npy/' + npy)
            y[i] = labels[npy]

        return X, y

# fit CheXNet

batch_size = 16
epochs = 200
# X_train1 = np.load(path+"train_5000_images.npy", mmap_mode = 'r')
# y_train1 = np.load(path+"train_5000_labels.npy", mmap_mode = 'r')
# X_valid = np.load(path+"valid_images.npy", mmap_mode = 'r')
# y_valid = np.load(path+"valid_labels.npy", mmap_mode = 'r')
# X_test = np.load(path+"test_images.npy", mmap_mode = 'r')
# y_test = np.load(path+"test_labels.npy", mmap_mode = 'r')

# chex_net.fit(X_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid), shuffle=True, callbacks=callbacks)
# chex_net.fit(X_train1, y_train1, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1, shuffle=True, callbacks=callbacks)

labels = {}
partition = {}
f = open(path + 'labels.pickle', 'rb')
labels = pickle.load(f)
f.close()
f = open(path + 'partition.pickle', 'rb')
partition = pickle.load(f)
f.close()

training_generator = DataGenerator(shape=(224,224), batch_size=batch_size).generate(labels, partition['train'])
validation_generator = DataGenerator(shape=(224,224), batch_size=batch_size).generate(labels, partition['validation'])

chex_net.fit_generator( generator=training_generator,
                        steps_per_epoch=len(partition['train'])//batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=len(partition['validation'])//batch_size )
