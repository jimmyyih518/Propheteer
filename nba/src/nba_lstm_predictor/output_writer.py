from typing import Any

from ..pipeline.pipeline_component import PipelineComponent


class NbaLstmPredictorOutputWriter(PipelineComponent):
    def __init__(self, component_name: str, input_key: str = None):
        super().__init__(component_name, input_key)

    def process(self, state: Any) -> None:
        # not implemented yet
        data = state.get(self.input_key)
        state.set(self.component_name, data)
