# -*- coding: utf-8 -*-

#  Imports
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import  Adam
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import preprocess_input
from DL_models import DeepMultimodal
from keras.layers import Dense, BatchNormalization, Activation, Input, concatenate
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D


def build_ssae(FC_len, firstL, secondL):
    
    input_shape = (FC_len,)
    inputs = Input(shape=input_shape)
    #    first SAE
    encoded = Dense(firstL)(inputs)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    #    second SAE
    encoded = Dense(secondL)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)

    #    inner SDE
    decoded = Dense(firstL)(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    #    outer SDE
    decoded = Dense(FC_len)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    
    ##   combine
    model = Model(inputs = inputs, 
                  outputs = decoded,
                  name = "SSAE")
    
    return model


##  create unlabeled synthetic samples as Cohort I

FC_features = np.random.rand(261,90,90,3)
SC_features = np.random.rand(261,90,90,3)


##  load VGG pretrained network
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(90,90,3))
TL_model = Model(inputs=base_model.input, outputs=base_model.output)
for layer in TL_model.layers:
      layer.trainable=False
      
##   dimension reduction with VGG to the same size as target cohort
FC_train = preprocess_input(FC_features)
SC_train = preprocess_input(SC_features)

FC_train = TL_model.predict(FC_train)
SC_train = TL_model.predict(SC_train)

##   flatten connectome
FC_train_flat = FC_train.reshape((FC_train.shape[0], FC_train.shape[1]*FC_train.shape[2]*FC_train.shape[3]))
SC_train_flat = SC_train.reshape((SC_train.shape[0], SC_train.shape[1]*SC_train.shape[2]*SC_train.shape[3]))


##   train SSAE using cohort II in an unsupervised learning
##   the number of nodes in first and second layer should match to deep multimodal learning
##   using SC as an example 

model_ssae = build_ssae(SC_train_flat.shape[1], 64, 16) 

lr = 0.01
epoch_max = 5
my_optimizer = Adam(lr=lr, decay=lr/epoch_max)
model_ssae.compile(optimizer=my_optimizer,loss='mse',metrics=['accuracy'])  

    # Fit the model
model_training = model_ssae.fit(SC_train_flat, SC_train_flat,                                   
                               epochs=epoch_max,
                               verbose=1,
                               shuffle=False,
                               batch_size=8)    

ssae_weights = model_ssae.get_weights()

ssae_l1 = ssae_weights[0]
ssae_l2 = ssae_weights[6]

np.save('ssae_layer1.npy', ssae_l1)
np.save('ssae_layer2.npy', ssae_l2)

##  same process for FC
