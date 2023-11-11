from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Encapsulates the basic methods of a PyTorch based model

    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load_model(self):
        """
        Load the model from a file. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def predict(self):
        """
        Perform a prediction, usually calling the forward method. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def train(self):
        """
        Perform a training run. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save_model(self):
        """
        Save the model to file. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def forward(self):
        """
        Standard forward pass for PyTorch Models. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
