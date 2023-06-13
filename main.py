from model import distillated_model, CNN_Mnist
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
y_train = y_train.astype(int)
y_test = y_test.astype(int)


distillation = distillated_model(temperature=25.0)

# Compile the model with appropriate loss and optimizer
distillation.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Train the model on the training data
distillation.fit(x_train, y_train, epochs=25, batch_size=32)

# Make predictions using the trained model+
probabilities = distillation.predict(x_train)

cnn = CNN_Mnist(temperature=1.0)

cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

cnn.fit(x_train,probabilities,epochs=25,batch_size=32)

test_probabilities = distillation.predict(x_test)

test_loss, test_acc = cnn.evaluate(x_test, test_probabilities)

cnn.save('defensive_distillation.h5')