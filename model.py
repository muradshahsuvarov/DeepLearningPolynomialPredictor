import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

def get_y(x1, x2):   
    y = x1**2 + 2 * x1 * x2 + x2**2 
    return y
    

x1 = np.float32(np.random.randint(0,10, size=(1, 10000)))
x2 = np.float32(np.random.randint(0,10, size=(1, 10000)))
y = get_y(x1, x2)
print(y)


training_X1, test_X1 = x1[:,:8000], x1[:,2000:]
training_X2, test_X2 = x2[:,:8000], x2[:,2000:]
training_Y, test_Y = y[:,:8000], y[:,2000:]


training_X1 = training_X1 / 9
training_X2 = training_X2 / 9

test_X1 = test_X1 / 9
test_X2 = test_X2 / 9

training_Y = training_Y



print(training_X1.shape)
print(training_X2.shape)
print(training_Y.shape)



training_X1


training_X2


merged_train = np.stack([training_X1, training_X2], axis=-1)


merged_train.shape
merged_train = merged_train[0, :, :]



merged_train



merged_train.shape



model = keras.Sequential([
keras.layers.Dense(2, input_dim=2, activation=keras.activations.tanh, use_bias=True),
keras.layers.Dense(40, input_dim=2, activation=keras.activations.tanh, use_bias=True),
keras.layers.Dense(40, input_dim=2, activation=keras.activations.tanh, use_bias=True),
keras.layers.Dense(1, activation=keras.activations.relu, use_bias=True),
])




model.compile(optimizer = 'adam',
             loss = 'mean_absolute_percentage_error',
             metrics=['accuracy'])
             
             
             
training_Y.T



model.fit(merged_train, training_Y.T, epochs=2000)



merged_test = np.stack([test_X1, test_X2], axis=-1)
merged_test.shape
merged_test = merged_test.reshape((8000,2))

test_loss, test_acc = model.evaluate(merged_test, test_Y.T, verbose = 1)



print(test_X1)
print(test_X2)
print(test_Y)
predictions = model.predict(merged_test)



print("------------Predicted------------")
print(predictions)
print("------------Predicted------------")
print("------------Real------------")
print(test_Y.T)
print("------------Real------------")


model.save(r'C:\Users\Murad\OneDrive\Рабочий стол\Deep Learning\SavedModels\EquationDLModel')