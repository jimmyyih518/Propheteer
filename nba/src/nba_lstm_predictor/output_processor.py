import pandas as pd
from ..pipeline.pipeline_component import PipelineComponent


class NbaLstmPredictorOutputProcessor(PipelineComponent):
    def __init__(self, component_name, input_key=None):
        super().__init__(component_name, input_key)

    def process(self, state):
        data = state.get(self.input_key)
        processed_data = self._process_output(data, state)
        state.set(self.component_name, processed_data)

    def _process_output(self, data, state):
        predictions = data["predictions"]
        original_targets = data["original_targets"]
        return {
            "predictions": pd.DataFrame(predictions, columns=state.TARGET_FEATURES),
            "original_targets": pd.DataFrame(
                original_targets, columns=state.TARGET_FEATURES
            ),
        }
