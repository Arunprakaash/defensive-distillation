import tensorflow as tf

def softmax(x, temperature):
    x = x / temperature
    exp_x = tf.exp(x)
    softmax_x = exp_x / tf.reduce_sum(exp_x)
    return softmax_x

class distillated_model(tf.keras.Model):
    def __init__(self, temperature=40.0):
        super(distillated_model, self).__init__()
        self.temperature = temperature

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.fc2(x)
        probabilities = softmax(logits, temperature=self.temperature)
        return probabilities

class CNN_Mnist(tf.keras.Model):
    def __init__(self, temperature=1.0):
        super(CNN_Mnist, self).__init__()
        self.temperature = temperature

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        logits = self.fc2(x)
        probabilities = softmax(logits, temperature=self.temperature)
        return probabilities