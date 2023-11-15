import boto3
import pandas as pd
import io
from typing import List, Any, Optional

from ..pipeline.pipeline_component import PipelineComponent
from ..constants.box_score_target_features import BoxScoreTargetFeatures


class NbaLstmPredictorDataLoader(PipelineComponent):
    """
    Data loader component for NBA LSTM Predictor, responsible for loading and processing data.

    Attributes:
        REQUIRED_COLUMNS (List[str]): A list of required column names.
        ADDITIONAL_INPUT_COLUMNS (List[str]): A list of additional input column names.
        SEQUENCE_LENGTH (int): The length of the sequence for LSTM input.
    """

    REQUIRED_COLUMNS: List[str] = [
        "dreb_pct_prev_season",
        "REB",  # Can be padded as zeros
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
        "BLK",  # Can be padded as zeros
        "ast_prev_season",
        "STL",  # Can be padded as zeros
        "player_height",
        "reb_prev_season",
        "usg_pct_prev_season",
        "ast_pct_prev_season",
        "rest_days",
        "AST",  # Can be padded as zeros
        "PTS",  # Can be padded as zeros
        "day_of_year",
    ]
    ADDITIONAL_INPUT_COLUMNS: List[str] = ["PLAYER_NAME", "GAME_DATE_EST"]
    SEQUENCE_LENGTH: int = 5

    def __init__(self, component_name: str, input_key: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            component_name (str): The name of the component.
            input_key (Optional[str]): The input key for the data source. Defaults to None.
        """
        super().__init__(component_name, input_key)
        self.s3 = boto3.client("s3")

    def process(self, state: Any) -> None:
        """
        Process the input data.

        Args:
            state (Any): The state object containing pipeline state.
        """
        # Setup pipeline constants
        self._set_constants(state)
        # Load input data from CSV
        data = self.load_csv(self.input_key)
        self.logger.info(f"Loaded CSV, first few rows {data.head()}")
        state.set(self.component_name, data)

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file from a specified filepath.

        Args:
            filepath (str): The filepath to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        if filepath.startswith("s3://"):
            return self._load_csv_from_s3(filepath)
        else:
            return self._load_csv_from_local(filepath)

    def _load_csv_from_local(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file from a local filepath.

        Args:
            filepath (str): The local filepath to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        try:
            self.logger.info(f"reading csv from {filepath}")
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            self.logger.exception(f"Error loading csv from {filepath}, {e}")
            raise

    def _load_csv_from_s3(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file from an S3 filepath.

        Args:
            filepath (str): The S3 filepath to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        self.logger.info(f"Retrieving input csv file from s3 {filepath}")
        s3_path_parts = filepath.replace("s3://", "").split("/")
        bucket_name = s3_path_parts[0]
        object_key = "/".join(s3_path_parts[1:])
        # Get the object from S3
        response = self.s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = response["Body"].read().decode("utf-8")

        # Use StringIO to convert the CSV string to a file-like object so it can be read into a DataFrame
        return pd.read_csv(io.StringIO(csv_content))

    def _set_constants(self, state: Any) -> None:
        """
        Set constants into the state object.

        Args:
            state (Any): The state object containing pipeline state.
        """
        self.logger.info("Setting Constants into State")
        state.set("REQUIRED_COLUMNS", self.REQUIRED_COLUMNS)
        state.set("ADDITIONAL_INPUT_COLUMNS", self.ADDITIONAL_INPUT_COLUMNS)
        state.set("TARGET_FEATURES", BoxScoreTargetFeatures.list())
        state.set("SEQUENCE_LENGTH", self.SEQUENCE_LENGTH)
