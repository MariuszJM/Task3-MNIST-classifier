from abc import ABC, abstractmethod
import numpy as np


class DigitClassificationInterface(ABC):

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> int:
        """
        Predict the digit for the given input_data.

        The input_data should be a 28x28X1 numpy array representing an image of a handwritten digit.

        The output will be a single integer representing the predicted digit.

        Each implementing class should handle the necessary preprocessing of the input_data to match the
        specific input format requirements of the underlying model.
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

