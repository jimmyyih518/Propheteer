import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..pipeline.pipeline_component import PipelineComponent
from .nba_torch_dataset import NbaDataset


class NbaLstmPredictorSequenceProcessor(PipelineComponent):
    def __init__(self, component_name, input_key=None):
        super().__init__(component_name, input_key)

    def process(self, state, batch_size=128):
        data = state.get(self.input_key)
        # Process sequence data
        (
            sequence_data,
            sequence_labels,
            sequence_player_team_ids,
            sequence_opponent_team_ids,
            sequence_date_ids,
            sequence_country_ids,
        ) = self._process_sequences(data, state)

        torch_dataset = NbaDataset(
            sequence_data,
            sequence_labels,
            sequence_player_team_ids,
            sequence_opponent_team_ids,
            sequence_date_ids,
            sequence_country_ids,
        )
        torch_dataloader = DataLoader(
            torch_dataset, batch_size=batch_size, shuffle=False
        )
        state.set(self.component_name, torch_dataloader)

    def _process_sequences(self, data, state):
        grouped_data = self._sort_and_group_df(data)
        sequence_data = []
        sequence_labels = []
        sequence_player_team_ids = []
        sequence_opponent_team_ids = []
        sequence_date_ids = []
        sequence_country_ids = []
        for name, group in tqdm(grouped_data):
            seq, lab, pt_ids, ot_ids, date_ids, country_ids = self._extract_sequences(
                group, state.SEQUENCE_LENGTH, state.TARGET_FEATURES, state
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

    def _sort_and_group_df(self, df):
        df.sort_values(by=["PLAYER_NAME", "GAME_DATE_EST"], inplace=True)
        df_grouped_by_player = df.groupby("PLAYER_NAME")

        return df_grouped_by_player

    def _extract_sequences(self, group, N, target_features, state):
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

            sequences.append(seq[state.REQUIRED_COLUMNS].values)
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
