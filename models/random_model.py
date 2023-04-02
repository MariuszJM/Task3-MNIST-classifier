import numpy as np
from models.digit_classification_interface import DigitClassificationInterface


class RandomModel(DigitClassificationInterface):
    def predict(self, input_data: np.ndarray) -> int:
        input_data = input_data[9:19, 9:19]

        random_prediction = np.random.randint(0, 10)
        return random_prediction
