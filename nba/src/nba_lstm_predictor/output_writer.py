from ..pipeline.pipeline_component import PipelineComponent

class NbaLstmPredictorOutputWriter(PipelineComponent):
    def __init__(self, component_name, input_key=None):
        super().__init__(component_name, input_key)

    def process(self, state):
        # not implemented yet
        data = state.get(self.input_key)
        state.set(self.component_name, data)