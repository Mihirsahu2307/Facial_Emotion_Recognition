import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, BatchNormalization, Dense
import os


# model6 using the mobile_net as the base model, trained on Kaggle. Input = (224, 224)
# Currently, mobilenet based model is used, to use the Resnet50 based model, make appropriate changes by referring to the jupyter notebook

classes_num = 7
img_size = 224
Project_DIR = "" # Earlier it was set to local directory path
mobilenet_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')

model6 = Sequential()
model6.add(mobilenet_model)

model6.add(Flatten())
model6.add(BatchNormalization())

model6.add(Dense(classes_num, activation='softmax'))

# model6.summary()
model6.load_weights(os.path.join(Project_DIR, "model_mobilenet", "model_mobilenet_input_224_50_epochs.h5"))