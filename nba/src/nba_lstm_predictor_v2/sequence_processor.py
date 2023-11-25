import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Any, Optional, Tuple
from ..pipeline.pipeline_component import PipelineComponent
from .nba_torch_dataset import NbaDatasetV2


class NbaLstmPredictorSequenceProcessor(PipelineComponent):
    """
    Sequence processor for NBA LSTM Predictor, responsible for processing and preparing sequence data for the LSTM model.

    This component processes the input data into sequences suitable for LSTM training or prediction.
    """

    def __init__(self, component_name: str, input_key: Optional[str] = None, batch_size:int = 128):
        """
        Initialize the sequence processor.

        Args:
            component_name (str): The name of the component.
            input_key (Optional[str]): The input key for the data source. Defaults to None.
            batch_size (int): The batch size for the DataLoader. Defaults to 128.
        """
        super().__init__(component_name, input_key)
        self.batch_size = batch_size

    def process(self, state: Any) -> None:
        """
        Process the input data into sequences and create a DataLoader for the LSTM model.

        Args:
            state (Any): The state object containing pipeline state.
        """

        data = state.get(self.input_key)
        # Process sequence data
        (
            sequence_data,
            sequence_labels,
            sequence_player_ids,
        ) = self._process_sequences(data, state)

        torch_dataset = NbaDatasetV2(
            sequence_data,
            sequence_labels,
            sequence_player_ids,
        )
        torch_dataloader = DataLoader(
            torch_dataset, batch_size=self.batch_size, shuffle=False
        )
        state.set(self.component_name, torch_dataloader)

    def _process_sequences(
        self, data: pd.DataFrame, state: Any
    ) -> Tuple[np.ndarray, ...]:
        """
        Process the DataFrame into sequences for LSTM input.

        Args:
            data (pd.DataFrame): The input data DataFrame.
            state (Any): The state object containing pipeline state.

        Returns:
            Tuple[np.ndarray, ...]: A tuple containing arrays of sequence data, labels, and various IDs.
        """

        grouped_data = self._sort_and_group_df(data)
        sequence_data = []
        sequence_labels = []
        sequence_player_ids = []
        for name, group in tqdm(grouped_data):
            seq, lab, pt_ids = self._extract_sequences(
                group, state.SEQUENCE_LENGTH, state.TARGET_FEATURES, state
            )
            sequence_data.extend(seq)
            sequence_labels.extend(lab)
            sequence_player_ids.extend(pt_ids)

        sequence_data = np.array(sequence_data)
        sequence_labels = np.array(sequence_labels)
        sequence_player_ids = np.array(sequence_player_ids)

        return (
            sequence_data,
            sequence_labels,
            sequence_player_ids,
        )

    def _sort_and_group_df(self, df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
        """
        Sort and group the DataFrame by player name and game date.

        Args:
            df (pd.DataFrame): The DataFrame to sort and group.

        Returns:
            pd.core.groupby.DataFrameGroupBy: The grouped DataFrame.
        """

        df.sort_values(by=["PLAYER_NAME", "GAME_DATE_EST"], inplace=True)
        df_grouped_by_player = df.groupby("PLAYER_NAME")

        return df_grouped_by_player

    def _extract_sequences(
        self, group: pd.DataFrame, N: int, target_features: list, state: Any
    ) -> Tuple[list, ...]:
        """
        Extract sequences from the grouped DataFrame.

        Args:
            group (pd.DataFrame): The grouped DataFrame.
            N (int): The length of the sequence.
            target_features (list): The list of target features.
            state (Any): The state object containing pipeline state.

        Returns:
            Tuple[list, ...]: A tuple containing lists of sequences, labels, and various IDs.
        """

        sequences = []
        labels = []
        player_ids = []

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
            player_ids.append(seq["player_name_encoded"].values)

        return (
            sequences,
            labels,
            player_ids,
        )
