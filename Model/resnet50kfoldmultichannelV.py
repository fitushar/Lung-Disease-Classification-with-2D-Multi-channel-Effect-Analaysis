# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 16:30:00 2018
##important

This Script is for Experimenting with the multiple HU units and Different channels
About Input:
very import ****** Images should have same name in different folders so that there combination when marging them together is accurate.

1.change the folder name given line 71 according to your first channel image
  change the folder name given line 72 according to your second channel image
  change the folder name given line 71 according to your Third channel image

which will read 3 different image as img1 , img2 and img3
     line 113 will thn marge the image
     img5 = np.dstack((img1,img2,img3))

using this line you can tell the model which image you want to use for the classification

2.give the number to the labels 0 and 1.
in line 127 and 128.

###Output::

model saves the model, weights and normalizer for all the 4 folds.

@author: Fakrul-IslamTUSHAR
"""

##Python
import numpy as np
import os
import cv2
#import scikitplot as skplt
import h5py


###Keras
import keras
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.layers import MaxPooling2D
from keras.models import Model
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

###Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

#import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore
from sklearn.model_selection import KFold
from keras.layers import GlobalMaxPooling2D

from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

# =============================================================================
# Load Image
# =============================================================================
##Loading the data
PATH=os.getcwd()

##Giving the path of the directory
data_path1=PATH+'/train_1_Duke_Complete_range'
data_path2=PATH+'/train_2_Duke_HIGH'
data_path3=PATH+'/train_3_Duke_LOW'
##
data_dir_list=os.listdir(data_path1)##getting the list of the folder in the directory.
print(data_dir_list)
#
img_data_list=[] #making list to store all the images
#
for dataset in data_dir_list:
    img_list=os.listdir(data_path1+'/'+dataset) #taking the list of the images in each folder
    print('Loaded the image of dataset-'+ '{}\n'.format(dataset))

    for img in img_list:
        img_path1=data_path1+'/'+dataset+'/'+img #getting the image path
        img1 = cv2.imread((img_path1),-1)
        img1 = cv2.resize(img1,(224,224))
# =============================================================================
#       getting 2nd one channel image
# =============================================================================
        img_path2=data_path2+'/'+dataset+'/'+img
        img2 = cv2.imread((img_path2),-1)
        img2 = cv2.resize(img2,(224,224))
# =============================================================================
#        getting 3rd One Channel Image
# =============================================================================
        img_path3=data_path2+'/'+dataset+'/'+img
        img3 = cv2.imread((img_path3),-1)
        img3 = cv2.resize(img3,(224,224))
        ###Marging the Image
        img5 = np.dstack((img1,img2,img3))
        #cv2.imwrite("original2.png",img5)
        #print('input image shape:',img2.shape)
        #cv2.imwrite("original2.png",img2)
        x=image.img_to_array(img5) #converting the image to array
        x=np.expand_dims(x,axis=0) #putting them in the row axis
        x=preprocess_input(x) #preprocessing
        print('input image shape:',x.shape)
        #cv2.imwrite("Final11.png",img2)
        img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


num_classes=1
print('Number of classes=',num_classes)
num_of_samples=img_data.shape[0]
print('Number of samples=',num_of_samples)
labels=np.ones((num_of_samples,),dtype='int64')
#labels=np.ones((num_of_samples,),dtype='float32')

#labels[0:1825]=0
#labels[1825:3127]=1

labels[0:701]=1
labels[701:1651]=0



#names=['non_mass','mass']

#now we need to conver the labels to on-hot encoding
#Y=np_utils.to_categorical(labels,num_classes)

#Suufle the data
x,y=shuffle(img_data,labels,random_state=2)

#split the training and test set
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.50,random_state=2)
#print('Training data:',x_train.shape,'testing data=',x_test.shape)
#print('Training labels:',y_train.shape,'testing data=',y_test.shape)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value




def get_model(net,num_classes=2):

    if net=='Resnet50':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.resnet50.ResNet50(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='softmax', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            #Freezing the upper layers
            #for layer in custom_model.layers[:-2]:
                #layer.trainable=False

            #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            custom_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy',auc_roc,f1_score])
            return custom_model

    elif net=='Vgg16':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.vgg16.VGG16(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='softmax', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            custom_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy',auc_roc])
            return custom_model

    elif net=='InceptionV3':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.inception_v3.InceptionV3(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='softmax', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            #Freezing the upper layers
            #for layer in custom_model.layers[:-1]:
                #layer.trainable=False

            custom_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy',auc_roc])
            return custom_model

    elif net=='DenseNet':
            image_input=Input(shape=(224,224,3))
            model=keras.applications.densenet.DenseNet121(input_tensor=image_input,include_top=False,weights='imagenet')
            last_layer = model.output
            z= keras.layers.GlobalMaxPooling2D()(last_layer)
            out = Dense(num_classes, activation='sigmoid', name='output_layer')(z)
            custom_model = Model(inputs=image_input,outputs= out)

            #Freezing the upper layers
            for layer in custom_model.layers[:-2]:
                layer.trainable=False

            custom_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy','acc'])
            return custom_model


# Cross-validate
kf = KFold(4)

oos_y = []
oos_pred = []
fold = 0

for train, test in kf.split(x):
    fold+=1
    print("Fold #{}".format(fold))

    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]



    datagen = ImageDataGenerator(
              featurewise_center=False,
              featurewise_std_normalization=False,
              rotation_range=0,
              vertical_flip=False)
    datagen.fit(x_train)

    batch_size=10

    custom_resnet_model=get_model('Resnet50')

    ###Check-point
    filepath= 'weights-best_model-Fold-{}'.format(fold)+'.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]

    custom_resnet_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),validation_data=(x_test, y_test),epochs=50, verbose=1, workers=4,callbacks=callbacks_list)


    pred = custom_resnet_model.predict(x_test)
    score= custom_resnet_model.evaluate(x_test, y_test)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    oos_y.append(y_test)
    oos_pred.append(pred)

    ###############################
    normalized_Xtest = datagen.standardize(x_test)
    y_pred_keras = custom_resnet_model.predict(normalized_Xtest, verbose=1, batch_size=batch_size)
    fpr_keras, tpr_keras,thresholds= roc_curve(y_test, y_pred_keras)
    train_auc = roc_auc_score(y_test, y_pred_keras)
    print("AUC: {}".format(train_auc))

    ####Model and Normalizer
    model_name = 'Model-Fold-{}'.format(fold)+'.json'
    normalizer_name = 'Normalizer-Fold-{}'.format(fold)+'.pkl'
    #####Saving t he model##########
    model_json = custom_resnet_model.to_json()
    with open(model_name, 'w') as json_file:
        json_file.write(model_json)

    joblib.dump(datagen,normalizer_name)

    print("model and weights have been saved")


    # Measure this fold's RMSE
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    print("Fold score (RMSE): {}".format(score))


# Build the oos prediction list and calculate the error.
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
score2 = np.sqrt(metrics.mean_squared_error(oos_pred,oos_y))
print("Final, out of sample score (RMSE): {}".format(score2))
