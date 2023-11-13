import pandas as pd
from abc import ABC, abstractmethod


class BaseDataProcessor(ABC):
    def __init__(self):
        pass

    @classmethod
    def load_csv(self, filepath):
        return pd.read_csv(filepath)

    @abstractmethod
    def process_data(self, data):
        """
        Process data. This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
