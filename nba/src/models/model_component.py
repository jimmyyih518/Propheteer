from ..pipeline.pipeline_component import PipelineComponent


class ModelComponent(PipelineComponent):
    def __init__(self, model, component_name, input_key=None):
        super().__init__(component_name, input_key)
        self.model = model


    def process(self, state):
        input_data = state.get(self.input_key)
        predictions = self.model.predict(input_data)
        state.set(self.component_name, predictions)
