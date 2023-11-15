import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Optional, Any


class NbaDataset(Dataset):
    """
    A PyTorch Dataset for handling NBA data.

    Attributes:
        X (List[torch.Tensor]): The input features.
        Y (List[torch.Tensor]): The target labels.
        player_team_ids (List[torch.Tensor]): Player team IDs.
        opponent_team_ids (List[torch.Tensor]): Opponent team IDs.
        date_ids (List[torch.Tensor]): Date IDs.
        country_ids (List[torch.Tensor]): Country IDs.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        player_team_ids: np.ndarray,
        opponent_team_ids: np.ndarray,
        date_ids: np.ndarray,
        country_ids: np.ndarray,
    ) -> None:
        """
        Initializes the NbaDataset with the given data.

        Args:
            X (np.ndarray): Input features.
            Y (np.ndarray): Target labels.
            player_team_ids (np.ndarray): Player team IDs.
            opponent_team_ids (np.ndarray): Opponent team IDs.
            date_ids (np.ndarray): Date IDs.
            country_ids (np.ndarray): Country IDs.
        """
        self.X = self.process_sequence_to_torch(X, torch.float32, np.float32)
        self.Y = self.process_sequence_to_torch(Y, torch.float32, np.float32)
        self.player_team_ids = self.process_sequence_to_torch(
            player_team_ids, torch.long, np.int64
        )
        self.opponent_team_ids = self.process_sequence_to_torch(
            opponent_team_ids, torch.long, np.int64
        )
        self.date_ids = self.process_sequence_to_torch(date_ids, torch.long, np.int64)
        self.country_ids = self.process_sequence_to_torch(
            country_ids, torch.long, np.int64
        )
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves the item at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the features and labels at the given index.
        """
        return (
            self.X[idx],
            self.Y[idx],
            self.player_team_ids[idx],
            self.opponent_team_ids[idx],
            self.date_ids[idx],
            self.country_ids[idx],
        )

    def convert_to_numeric(
        self, array: np.ndarray, num_type: Any = np.float32
    ) -> Optional[np.ndarray]:
        """
        Converts an array to the specified numeric type.

        Args:
            array (np.ndarray): The array to convert.
            num_type (data-type): The desired data type (default: np.float32).

        Returns:
            np.ndarray or None: The converted array or None if conversion fails.
        """
        try:
            return array.astype(num_type)
        except ValueError as e:
            self.logger.warning("Conversion error:", e)
            non_numeric = np.argwhere(
                ~pd.to_numeric(array.ravel(), errors="coerce").notna()
            )
            for idx in non_numeric:
                self.logger.warning(f"Non-numeric value at {idx}: {array.ravel()[idx]}")
            return None

    def process_sequence_to_torch(
        self,
        seq: np.ndarray,
        torchtype: torch.dtype = torch.float32,
        num_type: Any = np.float32,
    ) -> List[torch.Tensor]:
        """
        Processes a sequence of numpy arrays and converts them to PyTorch tensors.

        Args:
            seq (np.ndarray): The sequence of numpy arrays.
            torchtype (torch.dtype): The desired torch tensor type (default: torch.float32).
            num_type (data-type): The numpy data type for conversion (default: np.float32).

        Returns:
            List[torch.Tensor]: A list of PyTorch tensors.
        """
        output = []
        for x in tqdm(seq):
            if isinstance(x, np.ndarray):
                if x.dtype == np.object_:
                    x = self.convert_to_numeric(x, num_type)
                    if x is not None:
                        output.append(torch.tensor(x, dtype=torchtype))
                else:
                    output.append(torch.tensor(x, dtype=torchtype))
            else:
                self.logger.warning("Non-array entry:", x)
        return output


# Example usage
# dataset = NbaDataset(X_data, Y_data, player_team_ids, opponent_team_ids, date_ids, country_ids)
