import pandas as pd
from typing import Any, Optional, Dict

from ..pipeline.pipeline_component import PipelineComponent
from ..constants.model_modes import ModelRunModes


class NbaLstmPredictorOutputProcessor(PipelineComponent):
    """
    Output processor for NBA LSTM Predictor, responsible for processing the output of the model based on the run mode (predict or train).

    Attributes:
        MODEL_RUNMODE (str): A constant key for accessing model run mode in the state.
    """

    MODEL_RUNMODE: str = "MODEL_RUNMODE"

    def __init__(self, component_name: str, input_key: Optional[str] = None):
        """
        Initialize the output processor.

        Args:
            component_name (str): The name of the component.
            input_key (Optional[str]): The input key for the data source. Defaults to None.
        """

        super().__init__(component_name, input_key)

    def process(self, state: Any) -> None:
        """
        Process the output data based on the model mode (predict or train).

        Args:
            state (Any): The state object containing pipeline state.
        """

        data = state.get(self.input_key)
        if state.data[self.MODEL_RUNMODE] == ModelRunModes.predict.value:
            processed_data = self._process_prediction_output(data, state)
        elif state.data[self.MODEL_RUNMODE] == ModelRunModes.train.value:
            processed_data = self._process_training_output(data, state)
        else:
            raise ValueError(f"Input model mode not one of {ModelRunModes.list()}")

        state.set(self.component_name, processed_data)

    def _process_prediction_output(
        self, data: Dict[str, Any], state: Any
    ) -> Dict[str, pd.DataFrame]:
        """
        Process the prediction output data.

        Args:
            data (Dict[str, Any]): The data containing predictions and original targets.
            state (Any): The state object containing pipeline state.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing processed prediction data.
        """

        predictions = data["predictions"]
        original_targets = data["original_targets"]
        return {
            "predictions": pd.DataFrame(predictions, columns=state.TARGET_FEATURES),
            "original_targets": pd.DataFrame(
                original_targets, columns=state.TARGET_FEATURES
            ),
        }

    def _process_training_output(self, data: Any, state: Any) -> Dict[str, Any]:
        """
        Process the training output data.

        Args:
            data (Any): The data containing training and validation losses.
            state (Any): The state object containing pipeline state.

        Returns:
            Dict[str, Any]: A dictionary containing processed training data.
        """

        return {"model": data}
