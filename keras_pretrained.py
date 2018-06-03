# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 00:00:45 2018

@author: jaydeep thik
"""

import keras
from keras.applications import VGG16
from keras import layers, optimizers,models
from keras.preprocessing.image import ImageDataGenerator


conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))
conv_base.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, height_shift_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_genarator = train_datagen.flow_from_directory('F:\machine learning\code\cats_vs_dogs_small\\train', target_size=(150,150,3), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('F:\machine learning\code\cats_vs_dogs_small\\validation', target_size=(150,150,3), batch_size=20,class_mode='binary')

model.fit_generator(train_genarator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)

#fine tuning the parameters

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable=True
    if set_trainable:    
        layer.trainable = True
    else:
        layer.trainable = False
        

model.compile(optimizer=optimizers.RMSprop(lr=1e-5), loss='binary_crossentropy', metrics=['acc'])
model.fit_generator(train_genarator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
