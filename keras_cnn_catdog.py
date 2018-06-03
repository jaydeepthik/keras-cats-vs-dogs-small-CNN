# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 09:44:03 2018

@author: jaydeep thik
"""

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator 


model  = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (150,150,3)))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer= optimizers.RMSprop(1e-4), loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('F:\machine learning\code\cats_vs_dogs_small\\train', target_size=(150,150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('F:\machine learning\code\cats_vs_dogs_small\\validation', target_size=(150,150), batch_size=32, class_mode='binary')


history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
"""

train_datagen = ImageDataGenerator(rescale=1/255., rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, shear_range=0.2, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(dirname, batch_size=20, target_size=(150,150,3), class_mode='binary')
validation_generator = test_datagen.flow_from_directory(dirname, batch_size=20, target_size=(150,150,3), class_mode='binry')

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=100)
"""