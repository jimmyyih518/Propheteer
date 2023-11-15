import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from .pipeline_state import PipelineState


class PipelineComponent(ABC):
    """
    An abstract base class representing a component in a pipeline.

    This class provides a structure for a pipeline component, including methods for processing
    and identifying the component. It is designed to be subclassed to implement specific functionality
    for different types of pipeline components.

    Attributes:
        component_name (str): The name of the component.
        input_key (Optional[str]): The key for the input data used by this component. Defaults to None.
        logger (logging.Logger): Logger for the pipeline component.
    """

    def __init__(self, component_name: str, input_key: Optional[str] = None):
        """
        Initialize the PipelineComponent.

        Args:
            component_name (str): The name of the component.
            input_key (Optional[str]): The key for the input data this component should process. Defaults to None.
        """

        self.component_name = component_name
        self.input_key = input_key
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def process(self, state: PipelineState) -> None:
        """
        Abstract method to process the data in the pipeline state.

        This method should be implemented by subclasses to define the specific processing logic.

        Args:
            state (PipelineState): The shared pipeline state.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def component_name(self) -> str:
        """
        Get the unique name of the component.

        Returns:
            str: The name of the component.
        """

        return self.component_name

    def is_pipeline(self) -> bool:
        """
        Check if this component is a pipeline.

        This method can be overridden by subclasses that represent a pipeline.

        Returns:
            bool: False by default, indicating this component is not a pipeline.
        """

        return False
