from keras.models import load_model
import keras.backend as K
import cv2
import numpy as np

model_path = ''
img_path = ''

original_img = cv2.imraed(img_path)
width, height, _ = original_img.shape
img = cv2.resize(original_img, (224,224))

chexnet = load_model(model_path)

input_layer = chexnet.layers[0]
final_conv_layer = chexnet.layers[-4]
dense_layer = chexnet.layers[-2]
output_layer = chexnet.layers[-1]

class_weights = dense_layer.get_weights()
get_output = K.function(input_layer.input, [final_conv_layer.output, output_layer.output])
feature_maps, predictions = get_output(img)
feature_maps = feature_maps[0,:,:,:]

cam = np.zeros(shape=feature_maps.shape[:2], dtype=np.float32)

# for i, feature_map in enumera
