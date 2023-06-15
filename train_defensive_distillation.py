from model import ModelArchitecture, softmax
import tensorflow as tf
import load_mnist as mnist

#teacher model

distillation = ModelArchitecture()

distillation.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

distillation.fit(mnist.x_train, mnist.y_train, epochs=25, batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

soft_train_labels = softmax(distillation.predict(mnist.x_train),40.0)


# student model

cnn = ModelArchitecture()

cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

cnn.fit(mnist.x_train,soft_train_labels,epochs=25,batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

cnn.save_model('model/defensive_distillation/model')