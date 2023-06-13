import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

loaded_model = tf.saved_model.load('defensive_distillation')

# Load the MNIST test dataset
(_, _), (test_images, _) = mnist.load_data()

# Select a test image for prediction
test_image = test_images[0]
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image.astype(np.float32) / 255.0

# Perform the prediction using the loaded model
inputs = tf.convert_to_tensor(test_image)

predictions = loaded_model(inputs)

# Get the predicted label
predicted_label = np.argmax(predictions[0])

print("Predicted Label:", predicted_label)