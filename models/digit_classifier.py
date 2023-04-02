from models import classification_algorithms


class DigitClassifier:
    def __init__(self, algorithm):
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
