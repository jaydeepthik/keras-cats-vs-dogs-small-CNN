# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:54:27 2018

@author: jaydeep thik
"""

import numpy as np
import os, shutil

orig_dir = "F:\machine learning\Kaggle\catVsDog"
target_dir = "F:\machine learning\code\cats_vs_dogs_small"

train_dir =os.path.join(target_dir, 'train')
os.mkdir(train_dir)

test_dir =os.path.join(target_dir, 'test')
os.mkdir(test_dir)

val_dir =os.path.join(target_dir, 'validation')
os.mkdir(val_dir)

train_cats = os.path.join(train_dir, 'cats')
os.mkdir(train_cats)

train_dogs = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs)

test_cats = os.path.join(test_dir, 'cats')
os.mkdir(test_cats)

test_dogs = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs)

valid_dogs = os.path.join(val_dir, 'dogs')
os.mkdir(valid_dogs)

valid_cats = os.path.join(val_dir, 'cats')
os.mkdir(valid_cats)


names = ["cat.{}.jpg".format(i) for i in range(1000) ]

for name in names:
    source = os.path.join(orig_dir,'train\\'+name)
    des = os.path.join(train_cats,name)
    shutil.copy(source, des)

names = ["cat.{}.jpg".format(i) for i in range(1000,1500) ]

for name in names:
    source = os.path.join(orig_dir,'train\\'+name)
    des = os.path.join(test_cats,name)
    shutil.copy(source, des)

names = ["cat.{}.jpg".format(i) for i in range(1500,2000) ]

for name in names:
    source = os.path.join(orig_dir,'train\\'+name)
    des = os.path.join(valid_cats,name)
    shutil.copy(source, des)



names = ["dog.{}.jpg".format(i) for i in range(1000) ]

for name in names:
    source = os.path.join(orig_dir,'train\\'+name)
    des = os.path.join(train_dogs,name)
    shutil.copy(source, des)

names = ["dog.{}.jpg".format(i) for i in range(1000,1500) ]

for name in names:
    source = os.path.join(orig_dir,'train\\'+name)
    des = os.path.join(test_dogs,name)
    shutil.copy(source, des)

names = ["dog.{}.jpg".format(i) for i in range(1500,2000) ]

for name in names:
    source = os.path.join(orig_dir,'train\\'+name)
    des = os.path.join(valid_dogs,name)
    shutil.copy(source, des)

