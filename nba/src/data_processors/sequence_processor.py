import logging
import torch
import pickle
import pandas as pd
import numpy as np
from typing import List, Any
from tqdm import tqdm
from torch.utils.data import DataLoader
from .base_processor import BaseDataProcessor
from .nba_torch_dataset import NbaDataset
from ..constants.box_score_target_features import BoxScoreTargetFeatures

logger = logging.getLogger(__name__)


class NbaSequenceDataProcessor(BaseDataProcessor):
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
    TARGET_FEATURES: List[str] = BoxScoreTargetFeatures.list()
    SEQUENCE_LENGTH: int = 5

    def __init__(self, input_scaler_path, team_encoder_path, country_encoder_path):
        super().__init__()
        self.input_scaler = self._load_transform_function(input_scaler_path)
        self.team_encoder = self._load_transform_function(team_encoder_path)
        self.country_encoder = self._load_transform_function(country_encoder_path)

    def process_data(self, filepath, return_dataloader=True, batch_size=128):
        (
            sequence_data,
            sequence_labels,
            sequence_player_team_ids,
            sequence_opponent_team_ids,
            sequence_date_ids,
            sequence_country_ids,
        ) = self.process_data_sequence(filepath)
        torch_dataset = NbaDataset(
            sequence_data,
            sequence_labels,
            sequence_player_team_ids,
            sequence_opponent_team_ids,
            sequence_date_ids,
            sequence_country_ids,
        )

        if return_dataloader:
            return DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)
        else:
            return torch_dataset

    def process_data_sequence(self, filepath):
        csv_df = self.load_csv(filepath)
        self._validate_dataframe(csv_df)
        (
            sequence_data,
            sequence_labels,
            sequence_player_team_ids,
            sequence_opponent_team_ids,
            sequence_date_ids,
            sequence_country_ids,
        ) = ([], [], [], [], [], [])
        csv_df = self.process_dataframe_features(csv_df)
        csv_df = csv_df[self.ADDITIONAL_INPUT_COLUMNS+self.REQUIRED_COLUMNS]

        grouped_data = self._sort_and_group_df(csv_df)

        for name, group in tqdm(grouped_data):
            seq, lab, pt_ids, ot_ids, date_ids, country_ids = self._extract_sequences(
                group, self.SEQUENCE_LENGTH, self.TARGET_FEATURES
            )
            sequence_data.extend(seq)
            sequence_labels.extend(lab)
            sequence_player_team_ids.extend(pt_ids)
            sequence_opponent_team_ids.extend(ot_ids)
            sequence_date_ids.extend(date_ids)
            sequence_country_ids.extend(country_ids)

        sequence_data = np.array(sequence_data)
        sequence_labels = np.array(sequence_labels)
        sequence_player_team_ids = np.array(sequence_player_team_ids)
        sequence_opponent_team_ids = np.array(sequence_opponent_team_ids)
        sequence_date_ids = np.array(sequence_date_ids)
        sequence_country_ids = np.array(sequence_country_ids)

        return (
            sequence_data,
            sequence_labels,
            sequence_player_team_ids,
            sequence_opponent_team_ids,
            sequence_date_ids,
            sequence_country_ids,
        )

    def _validate_dataframe(self, df):
        required_columns = [
            col for col in self.REQUIRED_COLUMNS if col != "day_of_year"
        ] + self.ADDITIONAL_INPUT_COLUMNS
        if not all(column in df.columns for column in required_columns):
            logger.error(f"Input columns: {df.columns}")
            logger.error(f"Required columns: {required_columns}")
            missing = [x for x in required_columns if x not in df.columns]
            raise ValueError(f"Input Column Mismatch, missing: {missing}")

    def _sort_and_group_df(self, df):
        df.sort_values(by=["PLAYER_NAME", "GAME_DATE_EST"], inplace=True)
        df_grouped_by_player = df.groupby("PLAYER_NAME")

        return df_grouped_by_player

    def _extract_last_n_games(self, group, N):
        # If the group is empty, return an empty DataFrame of shape (N, len(group.columns))
        if group.empty:
            return pd.DataFrame(
                [[0] * len(group.columns)] * N, columns=group.columns
            ).reset_index(drop=True)

        # If less than N games, zero-pad the sequence; otherwise take last N games
        return (
            group[-N:].reset_index(drop=True)
            if len(group) >= N
            else pd.concat(
                [
                    pd.DataFrame(
                        [group.iloc[0]] * (N - len(group)), columns=group.columns
                    ),
                    group,
                ]
            ).reset_index(drop=True)
        )

    def _dropped_features(self, df, additional_features_to_drop=[]):
        return (
            ["PLAYER_NAME", "GAME_DATE_EST"]
            + [col for col in df if col.startswith("country_")]
            + [col for col in df if col.startswith("PREV_")]
            + additional_features_to_drop
        )

    def _extract_sequences(self, group, N, target_features):
        sequences = []
        labels = []
        player_team_ids = []
        opponent_team_ids = []
        date_ids = []
        country_ids = []

        for i in range(len(group)):
            if i < N:
                # Create a DataFrame with zero padding
                padded_zeros = pd.DataFrame(
                    [[0] * len(group.columns)] * (N - i - 1), columns=group.columns
                )
                seq = pd.concat([padded_zeros, group.iloc[: i + 1]])
            else:
                seq = group.iloc[i - N + 1 : i + 1]

            sequences.append(seq[self.REQUIRED_COLUMNS].values)
            labels.append(group.iloc[i][target_features].values)
            player_team_ids.append(seq["player_team_id"].values)
            opponent_team_ids.append(seq["opponent_team_id"].values)
            date_ids.append(seq["day_of_year"].values)
            country_ids.append(seq["country"].values)

        return (
            sequences,
            labels,
            player_team_ids,
            opponent_team_ids,
            date_ids,
            country_ids,
        )

    def process_dataframe_features(self, df):
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
            + self.TARGET_FEATURES
        ]
        df[numeric_features] = self.input_scaler.transform(df[numeric_features])
        for target in self.TARGET_FEATURES:
            df[target] = 0
        return df

    def _load_transform_function(self, function_path):
        with open(function_path, "rb") as file:
            transform_function = pickle.load(file)

        return transform_function
