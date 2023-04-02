import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.digit_classification_interface import DigitClassificationInterface


class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = RandomForestClassifier()
        return model

    def predict(self, input_data: np.ndarray) -> int:
        input_data = input_data.flatten().reshape(1, -1)
        predictions = self.model.predict(input_data)
        return predictions[0]
