import tensorflow as tf
import load_mnist as mnist
import numpy as np
import model as md

model = md.CNN_Mnist.load_model('model\cnn_mnist')
# FGSM attack parameters
epsilon = 0.1

# Generate adversarial examples using FGSM
x_test_adv = mnist.x_test + epsilon * np.sign(tf.GradientTape().gradient(model(mnist.x_test), mnist.x_test))

# Evaluate the model on normal and adversarial examples
_, acc_normal = model.evaluate(mnist.x_test, mnist.y_test, verbose=0)
_, acc_adv = model.evaluate(mnist.x_test_adv, mnist.y_test, verbose=0)

print(f"Accuracy on normal examples: {acc_normal * 100}%")
print(f"Accuracy on adversarial examples (FGSM attack): {acc_adv * 100}%")

print(model.predict(mnist.x_test[0]),mnist.y_test[0])
print(model.predict(x_test_adv[0]),mnist.y_test[0])
