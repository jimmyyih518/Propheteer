from abc import ABC, abstractmethod


class PipelineComponent(ABC):
    def __init__(self, component_name, input_key=None):
        self.component_name = component_name
        self.input_key = input_key

    @abstractmethod
    def process(self, state):
        """
        Process method for the component.

        :param state: The shared pipeline state.
        :param input_key: Optional key to specify which data to process. If None, use the output of the previous component.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def component_name(self):
        """
        A unique name for the component. Used for identifying output data in the pipeline state.
        """
        return self.component_name

    def is_pipeline(self):
        """
        Check if this component is a pipeline. Default is False, can be overridden by subclasses.
        """
        return False
