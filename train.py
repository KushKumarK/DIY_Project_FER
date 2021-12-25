import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

X_train = 'data/train'
X_valid = 'data/valid'

datagen_train = ImageDataGenerator(rescale=1./255)
datagen_valid = ImageDataGenerator(rescale=1./255)
train_generated = datagen_train.flow_from_directory(
    X_train,
    target_size= (56, 56),
    batch_size= 128,
    color_mode="gray_framescale",
    class_mode= "categorical"
)
valid_generator = datagen_valid.flow_from_directory(
    X_valid,
    target_size= (56, 56),
    batch_size= 128,
    color_mode= "gray_framescale",

)

