import logging
import boto3
import pandas as pd
import io
from typing import List, Any
from ..pipeline.pipeline_component import PipelineComponent
from ..constants.box_score_target_features import BoxScoreTargetFeatures


class NbaLstmPredictorDataLoader(PipelineComponent):
    REQUIRED_COLUMNS = [
        "dreb_pct_prev_season",
        "REB",
        "country",
        "draft_number",
        "draft_year",
        "age",
        "draft_round",
        "oreb_pct_prev_season",
        "player_weight",
        "player_team_at_home",
        "net_rating_prev_season",
        "opponent_team_id",
        "player_team_id",
        "ts_pct_prev_season",
        "gp_prev_season",
        "pts_prev_season",
        "SEASON",
        "BLK",
        "ast_prev_season",
        "STL",
        "player_height",
        "reb_prev_season",
        "usg_pct_prev_season",
        "ast_pct_prev_season",
        "rest_days",
        "AST",
        "PTS",
        "day_of_year",
    ]
    ADDITIONAL_INPUT_COLUMNS = ["PLAYER_NAME", "GAME_DATE_EST"]
    SEQUENCE_LENGTH: int = 5

    def __init__(self, component_name, input_key=None):
        super().__init__(component_name, input_key)
        self.logger = logging.getLogger(__name__)
        self.s3 = boto3.client("s3")

    def process(self, state):
        # Setup pipeline constants
        self._set_constants(state)
        # Load input data from CSV
        data = self.load_csv(self.input_key)
        print("loaded data", data)
        state.set(self.component_name, data)

    def load_csv(self, filepath):
        if filepath.startswith("s3://"):
            return self._load_csv_from_s3(filepath)
        else:
            return self._load_csv_from_local(filepath)

    def _load_csv_from_local(self, filepath):
        try:
            self.logger.info(f"reading csv from {filepath}")
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded CSV, first few rows {df.head()}")
            return df
        except Exception as e:
            self.logger.exception(f"Error loading csv from {filepath}, {e}")
            raise

    def _load_csv_from_s3(self, filepath):
        self.logger.info(f"Retrieving input csv file from s3 {filepath}")
        s3_path_parts = filepath.replace("s3://", "").split("/")
        bucket_name = s3_path_parts[0]
        object_key = "/".join(s3_path_parts[1:])
        # Get the object from S3
        response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response["Body"].read().decode("utf-8")

        # Use StringIO to convert the CSV string to a file-like object so it can be read into a DataFrame
        return pd.read_csv(io.StringIO(csv_content))

    def _set_constants(self, state):
        self.logger.info("Setting Constants into State")
        state.set("REQUIRED_COLUMNS", self.REQUIRED_COLUMNS)
        state.set("ADDITIONAL_INPUT_COLUMNS", self.ADDITIONAL_INPUT_COLUMNS)
        state.set("TARGET_FEATURES", BoxScoreTargetFeatures.list())
        state.set("SEQUENCE_LENGTH", self.SEQUENCE_LENGTH)
