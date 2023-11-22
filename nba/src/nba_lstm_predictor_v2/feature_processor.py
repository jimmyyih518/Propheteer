import json
import pickle
import pandas as pd
from typing import Optional, Any

from ..pipeline.pipeline_component import PipelineComponent


class NbaLstmPredictorFeatureProcessor(PipelineComponent):
    """
    Feature processor component for NBA LSTM Predictor, responsible for processing and transforming data features.

    Attributes:
        input_scaler: The scaler function for input normalization.
        team_encoder: The encoder function for team data.
        country_encoder: The encoder function for country data.
    """

    def __init__(
        self,
        component_name: str,
        input_key: Optional[str] = None,
        input_scaler_path: Optional[str] = None,
        player_encoder_path: Optional[str] = None,
    ):
        """
        Initialize the feature processor.

        Args:
            component_name (str): The name of the component.
            input_key (Optional[str]): The input key for the data source. Defaults to None.
            input_scaler_path (Optional[str]): File path to the input scaler pickle object. Defaults to None.
            team_encoder_path (Optional[str]): File path to the team encoder pickle object. Defaults to None.
            country_encoder_path (Optional[str]): File path to the country encoder pickle object. Defaults to None.
        """

        super().__init__(component_name, input_key)
        self.input_scaler = self._load_transform_function(input_scaler_path)
        self.player_encoder = self._load_transform_function(player_encoder_path)

    def process(self, state: Any) -> None:
        """
        Process the input data.

        Args:
            state (Any): The state object containing pipeline state.
        """

        data = state.get(self.input_key)
        self._validate_dataframe(data, state)
        processed_data = self._process_dataframe_features(data, state)
        processed_data = processed_data[
            state.ADDITIONAL_INPUT_COLUMNS + state.REQUIRED_COLUMNS + state.TARGET_FEATURES + ['player_name_encoded']
        ]
        state.set(self.component_name, processed_data)

    def _validate_dataframe(self, df: pd.DataFrame, state: Any) -> None:
        """
        Validate if the DataFrame contains all required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            state (Any): The state object containing pipeline state.

        Raises:
            ValueError: If required columns are missing in the DataFrame.
        """

        required_columns = [
            col for col in state.REQUIRED_COLUMNS if col != "day_of_year"
        ] + state.ADDITIONAL_INPUT_COLUMNS
        if not all(column in df.columns for column in required_columns):
            self.logger.error(f"Input columns: {df.columns}")
            self.logger.error(f"Required columns: {required_columns}")
            missing = [x for x in required_columns if x not in df.columns]
            raise ValueError(f"Input Column Mismatch, missing: {missing}")

    def _process_dataframe_features(self, df: pd.DataFrame, state: Any) -> pd.DataFrame:
        """
        Process and transform the features of the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            state (Any): The state object containing pipeline state.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """

        df["player_name_encoded"] = self.player_encoder.transform(df["PLAYER_NAME"])
        df[state.REQUIRED_COLUMNS] = self.input_scaler.transform(
            df[state.REQUIRED_COLUMNS]
        )
        # Pad target features with zero for inference only mode
        for target in state.TARGET_FEATURES:
            if not target in df.columns:
                df[target] = 0
        return df

    def _load_transform_function(self, function_path: str) -> Any:
        if function_path.endswith(".pkl"):
            with open(function_path, "rb") as file:
                return pickle.load(file)

        elif function_path.endswith(".json"):
            return json_transform_function(function_path)

        else:
            raise NotImplementedError(
                f"function_path extension for {function_path} not implemented"
            )


class json_transform_function:
    def __init__(self, function_path):
        with open(function_path) as file:
            self.json_data = json.load(file)

    def transform(self, data):
        output = []
        for item in data:
            output.append(self.json_data[item])

        return pd.Series(output)
