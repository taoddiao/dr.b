from densenet121 import DenseNet
from cheX_net import CheXNet
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from tqdm import tqdm

weights_path = 'chexnet_init_weights_tf.h5'

# dense_net121 = DenseNet(reduction=0.5, classes=1000, weights_path=weights_path)

chex_net = CheXNet(reduction=0.5)
chex_net.load_weights(weights_path)

# for layer in tqdm(dense_net121.layers):
#     if layer.name not in ['fc6', 'prob']:
#         chex_net.get_layer(name=layer.name).set_weights(layer.get_weights())
#
# chex_net.save_weights('chexnet_init_weights_tf.h5')

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1)

def weighted_binary_crossentropy(y_true, y_pred):
    pos_num = tf.count_nonzero(y_true, dtype=tf.float32)
    total_num = tf.size(y_true, out_type=tf.float32)
    weights = tf.divide(total_num - pos_num, pos_num)
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=weights), axis=-1)


chex_net.compile(optimizer=adam, loss=weighted_binary_crossentropy, metrics=['accuracy'])
