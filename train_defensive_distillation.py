from model import ModelArchitecture, softmax
import tensorflow as tf
import load_mnist as mnist

#teacher model

initial_network = ModelArchitecture()

initial_network.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

initial_network.fit(mnist.x_train, mnist.y_train, epochs=25, batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

soft_train_labels = softmax(initial_network.predict(mnist.x_train),40.0)


# student model

distillated_network = ModelArchitecture()

distillated_network.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

distillated_network.fit(mnist.x_train,soft_train_labels,epochs=25,batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

distillated_network.save_model('model/defensive_distillation/model')