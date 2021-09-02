# -*- coding: utf-8 -*-
"""
@author: Hailong Li
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#  Imports
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import  Adam
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import preprocess_input
from DL_models import DeepMultimodal



##  create synthetic samples as Cohort II
FC_features = np.random.rand(72,90,90,3)
SC_features = np.random.rand(72,90,90,3)
DWMA_features = np.random.rand(72,10)
Clinical_features = np.random.rand(72,72)
labels = np.random.rand(72)>0.5
sub_idx = np.array((range(72)))

## separate training and testing
train_idx, test_idx = train_test_split(sub_idx, shuffle=False)


FC_train = FC_features[train_idx,:,:,:]
SC_train = SC_features[train_idx,:,:,:]
DWMA_train = DWMA_features[train_idx,:]
Clinical_train = Clinical_features[train_idx,:]
labels_train = to_categorical(labels[train_idx])

##  Balance and Augument data
##  apply Data_bal_augmentation function for imbalanced dataset
##  The provided function is able to balance the dataset between positie and negative group
##  augment the dataset x10

FC_test = FC_features[test_idx,:,:,:]
SC_test = SC_features[test_idx,:,:,:]
DWMA_test = DWMA_features[test_idx,:]
Clinical_test = Clinical_features[test_idx,:]
labels_test = to_categorical(labels[test_idx])



##  load different pretrained network
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(90,90,3))
TL_model = Model(inputs=base_model.input, outputs=base_model.output)
for layer in TL_model.layers:
      layer.trainable=False
      
    ##   TL with VGG
         #  TL for train 
FC_train = preprocess_input(FC_train)
SC_train = preprocess_input(SC_train)

      
FC_train = TL_model.predict(FC_train)
SC_train = TL_model.predict(SC_train)

         #  TL for test 
FC_test = preprocess_input(FC_test)
SC_test = preprocess_input(SC_test)

        
FC_test = TL_model.predict(FC_test)
SC_test = TL_model.predict(SC_test)
  
    

    
#==============================================================================
##============  construct CNN based on keras ============##
FC_size = FC_train.shape[2]
SC_size = SC_train.shape[2]
chanel_size = FC_train.shape[3]
dwma_len = DWMA_train.shape[1]
clin_len = Clinical_train.shape[1]

model = DeepMultimodal.build(FC_size, chanel_size, SC_size, dwma_len, clin_len) 
#                        16, third_kern_size, 32, four_kern_size)

##  It is more efficient to off-line pretain SSAE using 'unsupevised_pretraining.py' function to obtain SSAE weights
##  load pretrained SSAE weights and assign to deep multimodal learning model
##  here we use ssae_layer1 and 2 as an example. 
##  Note:  FC and SC must be trained separately
ssae_l1 = np.load('ssae_layer1.npy')
ssae_l2 = np.load('ssae_layer2.npy')

W_int = model.get_weights()

#   transfer SSAE weights. FC layer index [0,24]  SC layer index [2,26]
W_tl = W_int[:]
W_tl[0] = ssae_l1
W_tl[24] = ssae_l2
        
W_tl[2] = ssae_l1
W_tl[26] = ssae_l2
        
model.set_weights(W_tl)    


# Compile the model
lr = 0.01
epoch_max = 10
my_optimizer = Adam(lr=lr, decay=lr/epoch_max)
model.compile(optimizer=my_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])         


# Fit the model in a supervised learning
model_training = model.fit([FC_train, SC_train, DWMA_train, Clinical_train],
                            labels_train, 
                           epochs=epoch_max,
                           shuffle=False,
                           verbose=1,
                           batch_size=8)



#  predicted and true labels
predictions = model.predict([FC_test, SC_test, DWMA_test, Clinical_test])
pred_y = np.argmax(predictions, axis=1)
true_y = np.argmax(labels_test, axis=1)


#  evaluate Acc   
score = model.evaluate([FC_test, SC_test, DWMA_test, Clinical_test],labels_test)
print('Test accuracy is {}'.format(score[1]))     


