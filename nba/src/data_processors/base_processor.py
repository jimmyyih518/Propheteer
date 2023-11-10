from abc import ABC, abstractmethod


class BaseDataProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process_data(self, data):
        """
        Process data. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
