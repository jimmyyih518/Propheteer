from abc import ABC, abstractmethod


class BaseDataProcessor(ABC):
    def __init__(self, processor_name=None):
        self.processor_name = processor_name

    @abstractmethod
    def process(self, data):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def load_data(self, filepath):
        raise NotImplementedError("This method should be overridden by subclasses.")
