from models import classification_algorithms
from typing import Literal


class DigitClassifier:
    def __init__(self, algorithm: Literal['cnn', 'rf', 'rand']):
        if algorithm == 'cnn':
            self.model = classification_algorithms.ConvolutionalNeuralNetwork()
        elif algorithm == 'rf':
            self.model = classification_algorithms.RandomForestModel()
        elif algorithm == 'rand':
            self.model = classification_algorithms.RandomModel()
        else:
            raise ValueError("Invalid algorithm specified")

    def predict(self, image):
        return self.model.predict(image)

    def train(self):
        return self.model.train()