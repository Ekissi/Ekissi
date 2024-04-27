
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import h5py
# import pandas as pd
# import scipy
# from keras.models import load_model
# from keras import preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Input, concatenate, Conv2D, Reshape, \
    GlobalMaxPooling2D, Flatten, GlobalAveragePooling2D, AveragePooling2D, Lambda, MaxPooling2D


# from datagenerator import DataGenerator

class DataGenerator(keras.utils.Sequence):
    def _init_(self, batch_size, dim=(2448, 2048), n_channels=3,shuffle=False, preprocess=None, list_IDs=[], db_path=''):
        'Initialization'
        self.preprocess = preprocess
        self.list_IDs_temp = []
        self.dim = dim
        self.db_path = db_path
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def _len_(self):
        'Denotes the number of batches per epoch'
        return (len(self.list_IDs))

    def _getitem_(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(self.list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for j, ID in enumerate(list_IDs_temp):
            # img=sklearn.datasets.load_sample_image(self.db_path+ID)
            id = os.path.join(self.db_path, ID)
            img = image.load_img(id, target_size=(2448, 2048))
            img = image.img_to_array(img)
            img = self.preprocess(img)
            # x = np.expand_dims(img, axis=0)
            X[j, :, :, :] = img

        return X


# import skimage
def Base_Model(base, weights='imagenet', include_top=False, input_shape=(2448, 2048, 3)):
    if (base == 'resnet50'):
        return ResNet50(weights=weights, include_top=include_top, input_shape=input_shape)
    if (base == 'resnet50'):
        return VGG16(weights=weights, include_top=include_top, input_shape=input_shape)


base = 'resnet50'
# model = Base_Model(base, weights='imagenet', include_top=False, input_shape=(2448, 2048, 3))

# def Feature_Extraction(train_ids,val_ids,test_ids,base='vgg16'):
base_model = Base_Model(base, weights='imagenet', include_top=False, input_shape=(2448, 2048, 3))
x = base_model.layers[-1].output
x1 = GlobalAveragePooling2D()(x)
model = keras.Model(inputs=base_model.layers[0].output, outputs=x1)
print("Model archtecture's details:")
model.summary()

# for m in range(1,6):
fd = os.listdir('imagebatch')
aa = len(fd)
b = aa + 1
generator = DataGenerator(os.listdir('imagebatch'),'imagebatch')
features = np.empty((aa, 2048))
for i in range(1, b):
    # img=sklearn.datasets.load_sample_image(self.db_path+ID)
    id = os.path.join('imagebatch', 'image' + str(i) + '.jpg')
    img = image.load_img(id, target_size=(2448, 2048))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    f = model(img)
    features[i - 1, :] = f
h5f = h5py.File('features_resnet50_ok.h5', 'w')
h5f.create_dataset('features', data=features)
h5f.close()
# del feat_test

# features = model.predict(generator)
# np.save('features_vgg16.npy', features)

# for m in range(1,6):
# fd = os.listdir('training' + str(m))
# aa=len(fd)
#  b =aa+1
#  generator=DataGenerator(batch_size=1, dim=(2448,2048), n_channels=3,
#     shuffle=False,preprocess=preprocess_input,list_IDs=os.listdir('training' + str(m)),db_path='training' + str(m))
#  feat_train = np.empty((aa,2048))
# for  i in range(1,b):
# img=sklearn.datasets.load_sample_image(self.db_path+ID)
# id =os.path.join('training' + str(m), 'image'+str(i)+'.jpg')
# img = image.load_img(id, target_size=(2448,2048))
#    img = image.img_to_array(img)
# img = preprocess_input(img)
# img = np.expand_dims(img, axis=0)
# f = model(img)
# feat_train[i-1,:]=f
# h5f = h5py.File('Training_features_resnet50'+str(m)+'.h5', 'w')
# h5f.create_dataset('training_features', data=feat_train)
# h5f.close()
# del feat_train