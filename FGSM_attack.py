import tensorflow as tf
import load_mnist as mnist
import numpy as np
import model as md
import matplotlib.pyplot as plt

epsilon = 0.1
image = mnist.x_train[0]
label = mnist.y_train[0]
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Model without distillation

model = md.CNN_Mnist.load_model('model\model_without_distillation\cnn_mnist')
defensive_distillation = md.distillated_model.load_model('model\defensive_distillation\model')

# FGSM attack parameters

def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

perturbations = adversarial_pattern(image.reshape((1, 28, 28, 1)), label).numpy()
adversarial = image + perturbations * epsilon

plt.imshow(adversarial.reshape((28, 28, 1)))
plt.show()

print(labels[model.predict(image.reshape((1, 28, 28, 1))).argmax()])
print(labels[model.predict(adversarial).argmax()])
print(labels[defensive_distillation.predict(adversarial).argmax()])
