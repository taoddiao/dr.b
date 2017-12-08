from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import cv2
import numpy as np
from cheX_net import CheXNet

K.set_learning_phase(1)
img_name = '00000899_000'
model_path = 'weights_dir/chexnet_weights_epoch_023_val_loss_1.2186.hdf5'
img_path = '/home/qwerty/data/NIH/images/%s.png' % img_name

original_img = cv2.imread(img_path)
width, height, _ = original_img.shape
img = np.empty((1,224,224,3))
img[0,:,:,:] = cv2.resize(original_img, (224,224))
img[:,:,:,0] = (img[:,:,:,0] - 103.94) * 0.017
img[:,:,:,1] = (img[:,:,:,1] - 116.78) * 0.017
img[:,:,:,2] = (img[:,:,:,2] - 123.68) * 0.017
# img = cv2.resize(original_img, (224,224))

chex_net = CheXNet(reduction=0.5)
chex_net.load_weights(model_path)


input_layer = chex_net.layers[0] # data
final_conv_layer = chex_net.layers[-4]  # relu5_blk
dense_layer = chex_net.layers[-2] # fc6
output_layer = chex_net.layers[-1] # prob


class_weights = dense_layer.get_weights()[0] # (1024, 1)

get_output = K.function([input_layer.input], [final_conv_layer.output, output_layer.output])
feature_maps, predictions = get_output([img])
feature_maps = feature_maps[0,:,:,:] # (7, 7, 1024)
print(predictions)
# print(feature_maps.shape)

cam = np.zeros(shape=feature_maps.shape[:2], dtype=np.float32)

for i, w in enumerate(class_weights[:, 0]):
    cam += w * feature_maps[:,:,i]
cam -= np.min(cam)
cam /= np.max(cam)
cam = np.uint8(255*cam)
cam = cv2.resize(cam, (height, width))
cv2.imwrite(img_name+'cam.png', cam)
heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
cv2.imwrite(img_name+'heatmap.png', heatmap)
# img = heatmap*0.5 + original_img
img = cv2.addWeighted(heatmap, 0.1, original_img, 0.9, 0)
cv2.imwrite(img_name + '.png', original_img)
cv2.imwrite(img_name + '_heatmap.png', img)
result = heatmap * 0.3 + original_img * 0.5
cv2.imwrite(img_name + '_heatmap2.png', result)
