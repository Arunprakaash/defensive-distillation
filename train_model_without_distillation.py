from model import CNN_Mnist
import tensorflow as tf
import load_mnist as mnist


model = CNN_Mnist()

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(mnist.x_train, mnist.y_train, epochs=25, batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

model.save('cnn_mnist.h5',save_format='h5')