import pandas as pd
from ..pipeline.pipeline_component import PipelineComponent
from ..constants.model_modes import ModelRunModes


class NbaLstmPredictorOutputProcessor(PipelineComponent):
    MODEL_RUNMODE: str = "MODEL_RUNMODE"

    def __init__(self, component_name, input_key=None):
        super().__init__(component_name, input_key)

    def process(self, state):
        data = state.get(self.input_key)
        if state.data[self.MODEL_RUNMODE] == ModelRunModes.predict:
            processed_data = self._process_prediction_output(data, state)
        elif state.data[self.MODEL_RUNMODE] == ModelRunModes.train:
            processed_data = self._process_training_output(data, state)
        else:
            raise ValueError(f"Input model mode not one of {ModelRunModes.list()}")

        state.set(self.component_name, processed_data)

    def _process_prediction_output(self, data, state):
        predictions = data["predictions"]
        original_targets = data["original_targets"]
        return {
            "predictions": pd.DataFrame(predictions, columns=state.TARGET_FEATURES),
            "original_targets": pd.DataFrame(
                original_targets, columns=state.TARGET_FEATURES
            ),
        }

    def _process_training_output(self, data, state):
        train_loss = data.train_losses
        validation_loss = data.val_losses
        return {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
