#data set - https://keras.io/api/datasets/fashion_mnist/

import tensorflow as tf
import numpy as np 
from tensorflow import keras 

#printing stuff
import matplotlib.pyplot as plt 

#Load a pre-defined data set (70k 28 by 28)
fashion_mnist = keras.datasets.fashion_mnist

# Pull form data set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Show Data
#print(train_labels[0])
#print(train_images[0])
#plt.imshow(train_images[1], cmap='gray', vmin = 0, vmax =255)
#plt.show()

#Define NN
model = keras.Sequential([
    
    #makes single 784*1 layer
    keras.layers.Flatten(input_shape=(28,28)),
    
    #hidden layer is 128 deep
    keras.layers.Dense(128, activation=tf.nn.relu),
    
   
    #keras.layers.Dense(128, activation=tf.nn.relu),
    
    #output is 0-10 (each their own clothing)
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

#compile model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train
model.fit(train_images, train_labels, epochs=5)

#test
test_loss = model.evaluate(test_images, test_labels)


plt.imshow(test_images[734], cmap='gray', vmin=0, vmax=255)
plt.show()

print(test_labels[734])

#make prediction
predictions = model.predict(test_images)

print(predictions[734])

#print prediction
print(list(predictions[734]).index(max(predictions[734])))



print("Done!")














