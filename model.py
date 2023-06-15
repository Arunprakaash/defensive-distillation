import tensorflow as tf

def softmax(soft_train_labels, temperature=1.0):
    return tf.nn.softmax(soft_train_labels / temperature)

class ModelArchitecture(tf.keras.Model):
    def __init__(self):
        super(ModelArchitecture, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(units=10,activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        probabilities = self.fc2(x)
        return probabilities
    
    def save_model(self, filepath):
        # Save the model weights
        self.save_weights(filepath)

    @classmethod
    def load_model(cls, filepath):
        # Create an instance of the model
        model = cls()
        # Load the saved weights
        model.load_weights(filepath)
        return model