import boto3
import torch
import io
import json
from typing import Any, Optional

from ..pipeline.pipeline_component import PipelineComponent
from ..models.player_box_score_predictor import PlayerBoxScoreLSTM
from ..constants.model_modes import ModelRunModes


class NbaLstmPredictorModel(PipelineComponent):
    """
    NBA LSTM Predictor Model component, responsible for managing the lifecycle of an LSTM model including loading, training, and prediction.

    Attributes:
        model (PlayerBoxScoreLSTM): The LSTM model for player box score prediction.
        model_mode (ModelRunModes): The mode in which the model is run (e.g., predict, train).
    """

    def __init__(
        self,
        component_name: str,
        input_key: Optional[str] = None,
        model_path: Optional[str] = None,
        model_config_path: Optional[str] = None,
        model_mode: str = ModelRunModes.predict.value,
    ):
        """
        Initialize the model component.

        Args:
            component_name (str): The name of the component.
            input_key (Optional[str]): The input key for the data source. Defaults to None.
            model_path (Optional[str]): The file path to the trained model. Defaults to None.
            model_config_path (Optional[str]): The file path to the model configuration. Defaults to None.
            model_mode (ModelRunModes): The mode in which the model is run. Defaults to ModelRunModes.predict.
        """

        super().__init__(component_name, input_key)
        self.s3 = boto3.client("s3")

        with open(model_config_path) as f:
            model_config = json.load(f)

        self.model = PlayerBoxScoreLSTM(**model_config)
        self.logger.info(f"Loading model on initialization for {model_mode}")
        self.model_mode = model_mode
        self._load_model(model_path)

    def process(self, state: Any) -> None:
        """
        Process the input data based on the model mode (predict or train).

        Args:
            state (Any): The state object containing pipeline state.
        """

        data = state.get(self.input_key)
        state.set("MODEL_RUNMODE", self.model_mode)

        if self.model_mode == ModelRunModes.predict.value:
            processed_data = self._predict(data, state)
        elif self.model_mode == ModelRunModes.train.value:
            processed_data = self._train(data, state)
        else:
            raise ValueError(
                f"Input model mode {self.model_mode} not one of {ModelRunModes.list()}"
            )

        state.set(self.component_name, processed_data)

    def _load_model(self, model_path: str) -> None:
        """
        Load the model from the specified path.

        Args:
            model_path (str): The file path to the trained model.
        """

        if model_path.startswith("s3://"):
            self._load_model_from_s3(model_path)
        else:
            self._load_model_from_local(model_path)
        self.logger.info("Model completed loading")

    def _load_model_from_s3(self, model_path: str) -> None:
        """
        Load the model from an S3 path.

        Args:
            model_path (str): The S3 path to the trained model.
        """

        # Extract bucket name and object key from the model_path
        self.logger.info(f"Retrieving model artifacts from s3 {model_path}")
        s3_path_parts = model_path.replace("s3://", "").split("/")
        bucket_name = s3_path_parts[0]
        object_key = "/".join(s3_path_parts[1:])

        # Download the object
        obj = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        state_dict = torch.load(io.BytesIO(obj["Body"].read()))

        # Load state dict into the model
        self.model.load_state_dict(state_dict)

    def _load_model_from_local(self, model_path: str) -> None:
        """
        Load the model from a local file path.

        Args:
            model_path (str): The local file path to the trained model.
        """

        self.logger.info(f"Loading model artifacts from local file {model_path}")
        self.model.load_state_dict(torch.load(model_path))

    def _predict(self, data: Any, state: Any) -> dict:
        """
        Make predictions using the model.

        Args:
            data (Any): The input data for making predictions.
            state (Any): The state object containing pipeline state.

        Returns:
            dict: A dictionary containing predictions and original targets.
        """

        predictions, original_targets = self.model.predict(data)
        return {
            "predictions": predictions,
            "original_targets": original_targets,
        }

    def _train(self, data: Any, state: Any, **kwargs) -> PlayerBoxScoreLSTM:
        """
        Train the model with the given data.

        Args:
            data (Any): The training data.
            **kwargs: Additional keyword arguments for training.

        Returns:
            PlayerBoxScoreLSTM: The trained model.
        """

        # Additional training params
        # learning_rate=0.0001
        # epochs=1000
        self.model.train(train_loader=data, **kwargs)
        self.logger.info(f"Final Train Loss: {self.model.train_losses[-1]}")
        self.logger.info(f"Final Validation Loss: {self.model.val_losses[-1]}")
        return self.model
