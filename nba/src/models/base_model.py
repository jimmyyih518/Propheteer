from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self):
        """
        Load the model from a file. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def predict(self, data):
        """
        Perform a prediction. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
