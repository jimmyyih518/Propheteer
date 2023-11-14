import logging
import boto3
import torch
import io
from ..pipeline.pipeline_component import PipelineComponent
from ..models.player_box_score_predictor import PlayerBoxScoreLSTM


class NbaLstmPredictorModel(PipelineComponent):
    def __init__(self, component_name, input_key=None, model_path=None):
        super().__init__(component_name, input_key)
        self.logger = logging.getLogger(__name__)
        self.model = PlayerBoxScoreLSTM(
            input_size=28,
            output_size=5,
            dropout=0.2,
            max_hidden_size=128,
            team_embedding_dim=64,
            date_embedding_dim=64,
            country_embedding_dim=64,
            lstm_hidden_dim=128,
        )
        self.logger.info("Loading model on initialization")
        self._load_model(model_path)

    def process(self, state, input_key=None):
        data = state.get(input_key)
        processed_data = self.model.predict(data, state)
        state.set(self.component_name, processed_data)

    def _load_model(self, model_path):
        if model_path.startswith("s3://"):
            self._load_model_from_s3(model_path)
        else:
            self._load_model_from_local(model_path)
        self.logger.info("Model completed loading")

    def _load_model_from_s3(self, model_path):
        # Extract bucket name and object key from the model_path
        self.logger.info(f"Retrieving model artifacts from s3 {model_path}")
        s3_path_parts = model_path.replace("s3://", "").split("/")
        bucket_name = s3_path_parts[0]
        object_key = "/".join(s3_path_parts[1:])

        # Initialize S3 client
        s3 = boto3.client("s3")

        # Download the object
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        state_dict = torch.load(io.BytesIO(obj["Body"].read()))

        # Load state dict into the model
        self.model.load_state_dict(state_dict)

    def _load_model_from_local(self, model_path):
        self.logger.info(f"Loading model artifacts from local file {model_path}")
        self.model.load_state_dict(torch.load(model_path))

    def _predict(self, data, state):
        predictions, original_targets = self.model.predict(data)
        return {
            "predictions": predictions,
            "original_targets": original_targets,
        }
