from model import distillated_model, CNN_Mnist, softmax
import tensorflow as tf
import load_mnist as mnist

#teacher model

distillation = distillated_model()

distillation.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

distillation.fit(mnist.x_train, mnist.y_train, epochs=25, batch_size=64,validation_data=(mnist.x_test,mnist.y_test))

soft_train_labels = softmax(distillation.predict(mnist.x_train),40.0)


# student model

cnn = CNN_Mnist()

cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

cnn.fit(mnist.x_train,soft_train_labels,epochs=25,batch_size=64,validation_data=(mnist.x_test,mnist.y_test))


# cnn.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
#               optimizer=tf.keras.optimizers.Adam(),
#               metrics=['accuracy'])

# cnn.fit(x_train, y_train, batch_size=128, epochs=20, validation_data=(mnist.x_test, mnist.y_test))


cnn.save_model('model/defensive_distillation')