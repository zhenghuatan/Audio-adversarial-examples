## This file contains the code written and used for the experiments of the following paper:
#
# Saeid Samizade, Zheng-Hua Tan, Chao Shen and Xiaohong Guan, “Adversarial Example Detection by Classification for Deep Speech Recognition”, ICASSP 2020.
#
# The datasets for the experiments are available here: http://kom.aau.dk/~zt/online/adversarial_examples/
#
# This code includes both training and test phases, but does not include the feature extraction part. 
# MFCC features are assumed to be saved in separated files in .mat format under "path"
#
# The dependencies of the code are: 
#	1- scipy 
#	2- numpy 
#	3- os
#	4- sklearn
#	5- matplotlib 
#	6- Tensorflow
#	7- Keras
#==========================================================================

import scipy
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

DATA_PATH_A = "./dataset-A/"
DATA_PATH_B = "./dataset-B/"
DATA_PATH_AB = "./dataset-AB/"

def get_labels(path):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def save_data_to_array(path):
    labels, _, _ = get_labels(path)
    for label in labels:
        mfcc_vectors = []
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + label)]
        for wavfile in wavfiles:
            c_mfcc = np.zeros((698, 40)) # 698 is the maximum number of windows in our samples. We use zero padding.
            mfcc = scipy.io.loadmat(wavfile)
            mfcc_data = mfcc['fea']
            for i in range(np.size(mfcc_data, 0)):
                for j in range(np.size(mfcc_data, 1)):
                    c_mfcc[i, j] = mfcc_data[i, j]
            mfcc_vectors.append(c_mfcc)
            print(c_mfcc.shape, mfcc_data.shape)
        np.save(label+'-B'+'.npy', mfcc_vectors)


def get_train_test(split_ratio=0.75, random_state=42, path=DATA_PATH_A):

    labels, indices, _ = get_labels(path)
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        print(label)
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio),
                            random_state=random_state, shuffle=True)

#===============================================================================================================
X_A_train, X_A_test, y_A_train, y_A_test = get_train_test(split_ratio=0.75, random_state=42, path=DATA_PATH_A)
X_A_train = X_A_train.reshape(X_A_train.shape[0], 698, 40, 1)
X_A_test = X_A_test.reshape(X_A_test.shape[0], 698, 40, 1)

y_A_train_hot = to_categorical(y_A_train)
y_A_test_hot = to_categorical(y_A_test)
#===============================================================================================================

model = Sequential()
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', input_shape=(698, 40, 1)))
model.add(MaxPooling2D(pool_size=(1, 3)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(Conv2D(32, kernel_size=(2, 2), activation='selu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history_A2A = model.fit(X_A_train, y_A_train_hot, batch_size=100, epochs=100, verbose=1,
          validation_data=(X_A_test, y_A_test_hot))

model.save('Adversarial_Normal.h5')

acc_A2A = history_A2A.history['acc']
val_acc_A2A = history_A2A.history['val_acc']
loss_A2A = history_A2A.history['loss']
val_loss_A2A = history_A2A.history['val_loss']
epochs = range(1, len(acc_A2A) + 1)
plt.plot(epochs, acc_A2A, lw=1, color='green', label='Training acc')
plt.plot(epochs, val_acc_A2A, '--',  lw=1, color='red', label='Validation acc')
plt.title('Training (A) and validation (A) accuracy')
plt.legend()
plt.savefig('ACC_A2A.pdf')

plt.clf()
plt.plot(epochs, loss_A2A, lw=1, color='green', label='Training loss')
plt.plot(epochs, val_loss_A2A, '--',  lw=1, color='red', label='Validation loss')
plt.title('Training (A) and validation (A) loss')
plt.legend()
plt.savefig('LOSS_A2A.pdf')

loss_history_A2A = history_A2A.history['loss']
acc_history_A2A = history_A2A.history['acc']

numpy_loss_history_A2A = np.array(loss_history_A2A)
np.savetxt("loss_history_A2A.txt", numpy_loss_history_A2A, delimiter=",")
numpy_acc_history_A2A = np.array(acc_history_A2A)
np.savetxt("acc_history_A2A.txt", numpy_acc_history_A2A, delimiter=",")
