from typing import Any, Optional
from ..pipeline.pipeline_state import PipelineState


class NbaLstmPipelineState(PipelineState):
    """
    A custom pipeline state class for the NBA LSTM pipeline, extending the basic functionality of PipelineState.

    This class provides methods to get, set, and remove data from the state, as well as to represent the state as a string.
    """

    def __init__(
        self, state_name: Optional[str] = None, local_dir: Optional[str] = None
    ):
        """
        Initialize the NBA LSTM pipeline state.

        Args:
            state_name (Optional[str]): The name of the state. Defaults to None.
        """
        super().__init__(state_name, local_dir)

    def get(self, key: str) -> Any:
        """
        Retrieve a value from the state using a key.

        Args:
            key (str): The key for the value to retrieve.

        Returns:
            Any: The value associated with the given key.
        """
        return self.data[key]

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the state with the specified key.

        Args:
            key (str): The key to associate with the value.
            value (Any): The value to store in the state.
        """
        self.data[key] = value

    def remove(self, key: str) -> None:
        """
        Remove a value from the state using its key.

        Args:
            key (str): The key of the value to remove.
        """
        self.data.pop(key, None)

    def __str__(self) -> str:
        """
        Return a string representation of the state.

        Returns:
            str: A string representation of the state, including keys and data.
        """
        string_repr = (
            "State Keys: "
            + str(self.data.keys())
            + "\n"
            + "State Data: "
            + str(self.data)
        )
        return string_repr

    def __repr__(self) -> str:
        """
        Return a string representation of the state. This is the same as __str__.

        Returns:
            str: A string representation of the state.
        """
        return self.__str__()
