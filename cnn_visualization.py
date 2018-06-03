# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 01:27:28 2018

@author: jaydeep thik
"""

from keras import models, layers
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


#load the model that was trained 
model = load_model("F:\machine learning\code\cats_vs_dogs_small\deep-mmodel-1.h5")

#loading an image
img = image.load_img("F:/machine learning/code/cats_vs_dogs_small/test/cats/cat.1002.jpg", target_size=(150,150,3))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /=255.
print(img_tensor.shape) 

#extraction of output of the layers
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs = layer_outputs)

activations = activation_model.predict(img_tensor)
l1_act = activations[1]
print(l1_act.shape)

plt.imshow(l1_act[0,:,:,8], cmap='viridis')