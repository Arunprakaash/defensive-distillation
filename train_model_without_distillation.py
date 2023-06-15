from model import CNN_Mnist
import tensorflow as tf
import load_mnist as mnist


model = CNN_Mnist()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(mnist.x_train, mnist.y_train, epochs=25, batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

model.save_model('model/cnn_mnist')