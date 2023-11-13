import boto3
import torch
import io
import logging
import pandas as pd
from typing import List, Any
from .base_model import BaseModel
from .player_box_score_predictor import PlayerBoxScoreLSTM
from ..constants.box_score_target_features import BoxScoreTargetFeatures

logger = logging.getLogger(__name__)


class NbaPredictor(BaseModel):
    TARGET_FEATURES: List[str] = BoxScoreTargetFeatures.list()

    def __init__(self, model_path, data_processor):
        super().__init__(model_path)
        self.data_processor = data_processor
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
        logger.info("Loading model on initialization")
        self.load_model()

    def load_model(self):
        if self.model_path.startswith("s3://"):
            self.load_model_from_s3()
        else:
            self.load_model_from_local()
        logger.info("Model completed loading")

    def load_model_from_s3(self):
        # Extract bucket name and object key from the model_path
        logger.info(f"Retrieving model artifacts from s3 {self.model_path}")
        s3_path_parts = self.model_path.replace("s3://", "").split("/")
        bucket_name = s3_path_parts[0]
        object_key = "/".join(s3_path_parts[1:])

        # Initialize S3 client
        s3 = boto3.client("s3")

        # Download the object
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        state_dict = torch.load(io.BytesIO(obj["Body"].read()))

        # Load state dict into the model
        self.model.load_state_dict(state_dict)

    def load_model_from_local(self):
        logger.info(f"Loading model artifacts from local file {self.model_path}")
        self.model.load_state_dict(torch.load(self.model_path))

    def predict(self, data):
        processed_sequence_data = self.data_processor.process_data(data)
        predictions, original_targets = self.model.predict(processed_sequence_data)
        return {
            "predictions": pd.DataFrame(predictions, columns=self.TARGET_FEATURES),
            "original_targets": pd.DataFrame(
                original_targets, columns=self.TARGET_FEATURES
            ),
        }
