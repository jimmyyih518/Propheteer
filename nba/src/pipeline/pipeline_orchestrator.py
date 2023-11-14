import logging
from .pipeline_component import PipelineComponent


class PipelineOrchestrator(PipelineComponent):
    def __init__(self, name, input_key=None, components=[], return_last_output=True):
        super().__init__(name, input_key)
        self.components = components
        self.last_output_key = input_key
        self.return_last_output = return_last_output
        self.logger = logging.getLogger(__name__)

    def process(self, state):
        final_output = None
        try:
            for component in self.components:
                self.logger.info(
                    f"Processing step {component.component_name} using input {self.last_output_key}"
                )
                print(f"Processing step {component.component_name} using input {self.last_output_key}")
                # Process the component or nested pipeline
                component.input_key = self.last_output_key
                self._process_component(component, state)

            # Fetch the final output
            final_output = state.get(self.last_output_key)

        except Exception as error:
            # Handle any exceptions that occurred during processing
            print(f"An error occurred: {error}")
        finally:
            self.logger.info("Cleaning up pipeline state")
            # Cleanup: delete intermediate states to free up memory
            self._cleanup_state(state)

            if self.return_last_output:
                return final_output

    def _process_component(self, component, state):
        """
        Process an individual component or a nested pipeline.

        :param component: The component or nested pipeline to process.
        :param state: The shared pipeline state.
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

    def _cleanup_state(self, state):
        """
        Cleanup the state by deleting intermediate data.
        """
        for key in list(state.data.keys()):
            if key != self.last_output_key:
                state.data.pop(key, None)

    def is_pipeline(self):
        return True
