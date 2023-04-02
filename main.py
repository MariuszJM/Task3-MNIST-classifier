from models.digit_classifier import DigitClassifier
import numpy as np


if __name__ == "__main__":
    classifier = DigitClassifier(algorithm="cnn")
    simulated_image = np.random.rand(28, 28, 1)
    classifier.train()
    prediction = classifier.predict(simulated_image)
    print("Prediction:", prediction)