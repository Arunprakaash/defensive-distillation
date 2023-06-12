import tensorflow as tf
from tensorflow.keras import models

# Load the trained model
model = models.load_model('model_without_distillation.h5')

# Select an image for the attack
image = test_images[0]
true_label = test_labels[0]

# Set the epsilon value (perturbation magnitude)
epsilon = 0.1

# Convert the image to a tensor and reshape it to match the model's input shape
image = tf.convert_to_tensor(image.reshape(1, 28, 28, 1))

# Record the image's gradient with respect to the loss
with tf.GradientTape() as tape:
    tape.watch(image)
    predictions = model(image)
    loss = tf.keras.losses.sparse_categorical_crossentropy(true_label, predictions)

# Calculate the gradient of the loss with respect to the image
gradient = tape.gradient(loss, image)

# Normalize the gradient and compute the perturbation
signed_gradient = tf.sign(gradient)
perturbation = epsilon * signed_gradient

# Create the adversarial image by adding the perturbation to the original image
adversarial_image = image + perturbation

# Clip the adversarial image to ensure its pixel values are within the valid range
adversarial_image = tf.clip_by_value(adversarial_image, clip_value_min=0.0, clip_value_max=1.0)

# Get the predictions for the adversarial image
adversarial_predictions = model(adversarial_image)

# Print the true label, original predictions, and adversarial predictions
print("True Label:", true_label)
print("Original Predictions:", predictions.numpy())
print("Adversarial Predictions:", adversarial_predictions.numpy())
