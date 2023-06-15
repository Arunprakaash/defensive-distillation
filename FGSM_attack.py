import tensorflow as tf
import load_mnist as mnist
import numpy as np
import model as md
import matplotlib.pyplot as plt

epsilon = 0.1
#train-0,test-4
image = mnist.x_test[4]
label = mnist.y_test[4]
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

# Model without distillation

model = md.ModelArchitecture.load_model('model\model_without_distillation\cnn_mnist')
defensive_distillation = md.ModelArchitecture.load_model('model\defensive_distillation\model')

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

print(model.predict(image.reshape((1, 28, 28, 1))))
print(model.predict(adversarial))
print(defensive_distillation.predict(adversarial))

print('-----------------------------------------------------------')

print(labels[model.predict(image.reshape((1, 28, 28, 1))).argmax()])
print(labels[model.predict(adversarial).argmax()])
print(labels[defensive_distillation.predict(adversarial).argmax()])
