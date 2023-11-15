from typing import Any, Optional, List
import logging
from .pipeline_component import PipelineComponent
from .pipeline_state import PipelineState


class PipelineOrchestrator(PipelineComponent):
    """
    Orchestrates the execution of a series of pipeline components.

    This class manages the sequential processing of different components in a pipeline,
    handling the transfer of data between components and maintaining the overall state.

    Attributes:
        components (List[PipelineComponent]): A list of components to be processed in the pipeline.
        last_output_key (str): The key of the last output in the pipeline state.
        return_last_output (bool): Flag to determine if the last output should be returned after processing.
        logger (logging.Logger): Logger for the orchestrator.
    """

    def __init__(
        self,
        name: str,
        input_key: Optional[str] = None,
        components: List[PipelineComponent] = [],
        return_last_output: bool = True,
    ):
        super().__init__(name, input_key)
        self.components = components
        self.last_output_key = input_key
        self.return_last_output = return_last_output
        self.logger = logging.getLogger(__name__)

    def process(self, state: PipelineState) -> Optional[Any]:
        """
        Process all components in the pipeline sequentially.

        Args:
            state (PipelineState): The shared state object for the pipeline.

        Returns:
            Optional[Any]: The final output of the pipeline if return_last_output is True, otherwise None.
        """

        final_output = None
        try:
            for component in self.components:
                self.logger.info(
                    f"Processing step {component.component_name} using input {self.last_output_key}"
                )
                # Process the component or nested pipeline
                component.input_key = self.last_output_key
                self._process_component(component, state)

            # Fetch the final output
            final_output = state.get(self.last_output_key)

        except Exception as error:
            # Handle any exceptions that occurred during processing
            self.logger.exception(f"An error occurred: {error}")
        finally:
            self.logger.info("Cleaning up pipeline state")
            # Cleanup: delete intermediate states to free up memory
            self._cleanup_state(state)

            if self.return_last_output:
                return final_output

    def _process_component(
        self, component: PipelineComponent, state: PipelineState
    ) -> None:
        """
        Process an individual component or a nested pipeline.

        Args:
            component (PipelineComponent): The component or nested pipeline to process.
            state (PipelineState): The shared pipeline state.
        """

        if component.is_pipeline():
            # For a nested pipeline, process each component in it sequentially
            for sub_component in component.components:
                self._process_component(sub_component, state)
        else:
            # Process an individual component
            component.process(state)

        # Update the last output key to the current component's name
        self.last_output_key = component.component_name

    def _cleanup_state(self, state: PipelineState) -> None:
        """
        Cleanup the state by deleting intermediate data.

        Args:
            state (PipelineState): The shared pipeline state.
        """

        for key in list(state.data.keys()):
            if key != self.last_output_key:
                state.data.pop(key, None)

    def is_pipeline(self) -> bool:
        """
        Check if this component is a pipeline.

        Returns:
            bool: Always returns True for PipelineOrchestrator.
        """

        return True
