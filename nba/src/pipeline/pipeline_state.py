import logging
from abc import ABC, abstractmethod


class PipelineState(ABC):
    def __init__(self, state_name=None):
        self.logger = logging.getLogger(__name__)
        self.state_name = state_name
        self._init_state()

    def _init_state(self):
        self.logger.warning("State type not set, default to dictionary(json)")
        self.data = {}

    @abstractmethod
    def get(self, key, default=None):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def set(self, key, value, mode="set"):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def remove(self, key):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __getattr__(self, item):
        """
        Override attribute access to provide direct access to data keys.

        :param item: The attribute/key name being accessed.
        :return: The value associated with the key in the state data.
        """
        try:
            return self.data[item]
        except KeyError:
            self.logger.error(f"Key {item} not found in state data.")
            raise AttributeError(f"State attribute '{item}' not found.")
