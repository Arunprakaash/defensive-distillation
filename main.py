from model import distillated_model, CNN_Mnist, softmax
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Load the MNIST dataset from TensorFlow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the images to match the expected shape of the model
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Convert labels to integers
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)


#teacher model

distillation = distillated_model()

distillation.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

distillation.fit(x_train, y_train, epochs=25, batch_size=64,validation_data=(x_test,y_test))

soft_train_labels = softmax(distillation.predict(x_train),40.0)


# student model

cnn = CNN_Mnist()

cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

cnn.fit(x_train,soft_train_labels,epochs=25,batch_size=64,validation_data=(x_test,y_test))


# cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])

# cnn.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(x_test, y_test))


cnn.save_weights('defensive_distillation.h5')