import pickle
import logging
import pandas as pd

from ..pipeline.pipeline_component import PipelineComponent


class NbaLstmPredictorFeatureProcessor(PipelineComponent):
    def __init__(
        self,
        component_name,
        input_key=None,
        input_scaler_path=None,
        team_encoder_path=None,
        country_encoder_path=None,
    ):
        super().__init__(component_name, input_key)
        self.input_scaler = self._load_transform_function(input_scaler_path)
        self.team_encoder = self._load_transform_function(team_encoder_path)
        self.country_encoder = self._load_transform_function(country_encoder_path)
        self.logger = logging.getLogger(__name__)

    def process(self, state):
        data = state.get(self.input_key)
        self._validate_dataframe(data, state)
        processed_data = self._process_dataframe_features(data, state)
        processed_data = processed_data[state.ADDITIONAL_INPUT_COLUMNS+state.REQUIRED_COLUMNS]
        state.set(self.component_name, processed_data)

    def _validate_dataframe(self, df, state):
        required_columns = [
            col for col in state.REQUIRED_COLUMNS if col != "day_of_year"
        ] + state.ADDITIONAL_INPUT_COLUMNS
        if not all(column in df.columns for column in required_columns):
            self.logger.error(f"Input columns: {df.columns}")
            self.logger.error(f"Required columns: {required_columns}")
            missing = [x for x in required_columns if x not in df.columns]
            raise ValueError(f"Input Column Mismatch, missing: {missing}")

    def _process_dataframe_features(self, df, state):
        df["day_of_year"] = pd.to_datetime(df["GAME_DATE_EST"]).dt.dayofyear
        df["country"] = self.country_encoder.transform(df["country"])
        df["player_team_id"] = self.team_encoder.transform(df["player_team_id"])
        df["opponent_team_id"] = self.team_encoder.transform(df["opponent_team_id"])
        numeric_features = [
            x
            for x in df.columns
            if x
            not in [
                "country",
                "player_team_id",
                "opponent_team_id",
                "PLAYER_NAME",
                "GAME_DATE_EST",
                "SEASON",
                "day_of_year",
                "player_team_at_home",
            ]
            + state.TARGET_FEATURES
        ]
        df[numeric_features] = self.input_scaler.transform(df[numeric_features])
        # Pad target features with zero for inference only mode
        for target in state.TARGET_FEATURES:
            df[target] = 0
        return df

    def _load_transform_function(self, function_path):
        with open(function_path, "rb") as file:
            transform_function = pickle.load(file)

        return transform_function
