from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import layers, models
import numpy as np
from models.digit_classification_interface import DigitClassificationInterface


class RandomModel(DigitClassificationInterface):

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, input_data: np.ndarray) -> int:
        input_data = input_data[9:19, 9:19]
        random_prediction = np.random.randint(0, 10)
        return random_prediction


class ConvolutionalNeuralNetwork(DigitClassificationInterface):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, input_data: np.ndarray) -> int:
        input_data = input_data.reshape(1, 28, 28, 1)
        predictions = self.model(input_data, training=False)
        return np.argmax(predictions)


class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, input_data: np.ndarray) -> int:
        input_data = input_data.flatten().reshape(1, -1)
        predictions = self.model.predict(input_data)
        return predictions[0]
