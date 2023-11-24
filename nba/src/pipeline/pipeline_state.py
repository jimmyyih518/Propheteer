import logging
import uuid
from datetime import datetime
from typing import Optional, Any
from abc import ABC, abstractmethod


class PipelineState(ABC):
    DEFAULT_DIR: str = "s3"

    """
    An abstract base class representing the state of a pipeline.

    This class provides a structure for managing the state within a pipeline, including
    methods for getting, setting, and removing data. It is designed to be subclassed
    to implement these methods as per specific requirements.

    Attributes:
        logger (logging.Logger): Logger for the pipeline state.
        state_name (Optional[str]): The name of the state.
        data (dict): The internal dictionary to store state data.
    """

    def __init__(self, state_name: Optional[str] = None, local_dir=None):
        """
        Initialize the PipelineState.

        Args:
            state_name (Optional[str]): The name of the state. Defaults to None.
        """

        self.logger = logging.getLogger(__name__)
        self.state_name = state_name if state_name else ""
        self._init_state_id(local_dir)
        self._init_state()

    def _init_state(self, state_type=None) -> None:
        """
        Initialize the state data structure.

        This method sets up the initial data structure for the state. It can be overridden
        by subclasses to initialize a different type of data structure.
        """

        if not state_type:
            self.logger.warning("State type not set, default to dictionary(json)")
            self.data = {}
        else:
            raise NotImplementedError("State type not implemented")

    @abstractmethod
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Abstract method to get a value from the state.

        Args:
            key (str): The key for the value to retrieve.
            default (Optional[Any]): The default value to return if the key is not found. Defaults to None.

        Returns:
            Any: The value associated with the given key or the default value.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def set(self, key: str, value: Any, mode: str = "set") -> None:
        """
        Abstract method to set a value in the state.

        Args:
            key (str): The key to associate with the value.
            value (Any): The value to store in the state.
            mode (str): The mode of setting the value. Defaults to "set".

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def remove(self, key: str) -> None:
        """
        Abstract method to remove a value from the state.

        Args:
            key (str): The key of the value to remove.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """

        raise NotImplementedError("This method should be overridden by subclasses.")

    def __getattr__(self, item: str) -> Any:
        """
        Override attribute access to provide direct access to data keys.

        Args:
            item (str): The attribute/key name being accessed.

        Returns:
            Any: The value associated with the key in the state data.

        Raises:
            AttributeError: If the key is not found in the state data.
        """

        try:
            return self.data[item]
        except KeyError:
            self.logger.error(f"Key {item} not found in state data.")
            raise AttributeError(f"State attribute '{item}' not found.")

    def _init_state_id(self, local_dir) -> None:
        state_dir = local_dir if local_dir else self.DEFAULT_DIR
        state_id_components = [
            state_dir,
            self.state_name,
            str(uuid.uuid4()),
            datetime.now().isoformat(),
        ]
        self.state_id = "_".join([value for value in state_id_components if value])
        self.logger.info(f"Initialized pipeline ID: {self.state_id}")
