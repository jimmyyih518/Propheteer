import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple

from .base_model import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PlayerBoxScoreLSTM(nn.Module, BaseModel):

    """
    LSTM Implementation to predict player box score stats for a given game
    """

    # Static Variables
    SCALING_FACTORS: dict = {
        "PTS": 80,
        "REB": 30,
        "AST": 25,
        "STL": 10,
        "BLK": 10,
    }
    MAX_TEAM_COUNT: int = 30
    MAX_DAYS_IN_YEAR: int = 367  # Rounding up on Leap Year
    MAX_COUNTRY_COUNT: int = 82

    def __init__(
        self,
        input_size: int,
        max_hidden_size: int = 128,
        dropout: float = 0.2,
        output_size: int = 5,
        team_embedding_dim: int = 10,
        date_embedding_dim: int = 10,
        country_embedding_dim: int = 10,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize the PlayerBoxScoreLSTM model.

        Args:
        - input_size (int): The size of the input features.
        - max_hidden_size (int): The maximum size of the hidden layers.
        - dropout (float): The dropout rate for regularization.
        - output_size (int): The size of the output layer.
        - team_embedding_dim (int): The dimensionality of the team embeddings.
        - date_embedding_dim (int): The dimensionality of the date embeddings.
        - country_embedding_dim (int): The dimensionality of the country embeddings.
        - lstm_hidden_dim (int): The number of features in the hidden state of the LSTM.
        - lstm_layers (int): The number of layers in the LSTM.
        - verbose (bool): If True, enables verbose logging.

        The model consists of an LSTM network with fully connected layers and embeddings for teams, dates, and countries.
        """

        super(PlayerBoxScoreLSTM, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc1 = nn.Linear(
            input_size
            + 2 * team_embedding_dim
            + date_embedding_dim
            + country_embedding_dim,
            lstm_hidden_dim,
        )
        self.lstm = nn.LSTM(
            lstm_hidden_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True
        )
        self.fc2 = nn.Linear(lstm_hidden_dim, int(max_hidden_size / 2))
        self.fc3 = nn.Linear(int(max_hidden_size / 2), int(max_hidden_size / 2))
        self.fc4 = nn.Linear(int(max_hidden_size / 2), output_size)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.team_embedding = nn.Embedding(self.MAX_TEAM_COUNT, team_embedding_dim)
        self.date_embedding = nn.Embedding(self.MAX_DAYS_IN_YEAR, date_embedding_dim)
        self.country_embedding = nn.Embedding(
            self.MAX_COUNTRY_COUNT, country_embedding_dim
        )
        self.scale_factors = torch.tensor(
            [
                self.SCALING_FACTORS["PTS"],
                self.SCALING_FACTORS["REB"],
                self.SCALING_FACTORS["AST"],
                self.SCALING_FACTORS["STL"],
                self.SCALING_FACTORS["BLK"],
            ],
            dtype=torch.float32,
        ).to(self.device)
        self.verbose = verbose
        self.train_losses = []
        self.val_losses = []
        if self.verbose:
            logger.info(f"Instantiated PlayerBoxScoreLSTM model")

    def forward(
        self,
        x: Tensor,
        player_team_ids: Tensor,
        opponent_team_ids: Tensor,
        date_ids: Tensor,
        country_ids: Tensor,
    ) -> Tensor:
        """
        Forward pass of the PlayerBoxScoreLSTM model.

        Args:
        - x (Tensor): The input tensor containing the features.
        - player_team_ids (Tensor): Tensor of player team IDs for embedding.
        - opponent_team_ids (Tensor): Tensor of opponent team IDs for embedding.
        - date_ids (Tensor): Tensor of date IDs for embedding.
        - country_ids (Tensor): Tensor of country IDs for embedding.

        Returns:
        - Tensor: The output of the model after processing the input through the layers.
        """
        # Get embeddings
        player_team_embed = self.team_embedding(player_team_ids)
        opponent_team_embed = self.team_embedding(opponent_team_ids)
        date_embed = self.date_embedding(date_ids)
        country_embed = self.country_embedding(country_ids)

        # Concatenate embeddings
        x = torch.cat(
            [x, player_team_embed, opponent_team_embed, date_embed, country_embed],
            dim=2,
        )
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm(x)

        x = torch.relu(self.dropout(self.fc2(x)))
        x = torch.relu(self.dropout(self.fc3(x)))
        x = torch.relu(self.fc4(x))
        return x

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.01,
        epochs: int = 5000,
    ) -> None:
        """
        Train the model using the given training data loader and optional validation data loader.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (Optional[DataLoader]): DataLoader for the validation dataset. If None, validation is skipped.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs to train the model.

        This method updates the model's weights based on the training data and optionally evaluates the model on the validation data.
        """
        self.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        epochs = range(epochs) if self.verbose else tqdm(range(epochs))

        for epoch in epochs:
            total_train_loss = 0
            for (
                X_batch,
                Y_batch,
                player_team_ids_batch,
                opponent_team_ids_batch,
                date_ids_batch,
                country_ids_batch,
            ) in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                player_team_ids_batch = player_team_ids_batch.to(self.device)
                opponent_team_ids_batch = opponent_team_ids_batch.to(self.device)
                date_ids_batch = date_ids_batch.to(self.device)
                country_ids_batch = country_ids_batch.to(self.device)

                outputs = self.forward(
                    X_batch,
                    player_team_ids_batch,
                    opponent_team_ids_batch,
                    date_ids_batch,
                    country_ids_batch,
                )
                outputs_last_timestep = outputs[:, -1, :]
                loss = torch.sqrt(
                    criterion(outputs_last_timestep, Y_batch)
                )  # RMSE Loss for last step only
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()  # Accumulate the batch loss

            avg_train_loss = total_train_loss / len(
                train_loader
            )  # Compute average training loss for the epoch
            # Append the average training loss
            self.train_losses.append(avg_train_loss)

            # Compute validation loss if validation data is provided
            if val_loader:
                total_val_loss = 0
                for (
                    X_val,
                    Y_val,
                    player_team_ids_val,
                    opponent_team_ids_val,
                    date_ids_val,
                    country_ids_val,
                ) in val_loader:
                    X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
                    player_team_ids_val = player_team_ids_val.to(self.device)
                    opponent_team_ids_val = opponent_team_ids_val.to(self.device)
                    date_ids_val = date_ids_val.to(self.device)
                    country_ids_val = country_ids_val.to(self.device)

                    val_outputs = self.forward(
                        X_val,
                        player_team_ids_val,
                        opponent_team_ids_val,
                        date_ids_val,
                        country_ids_val,
                    )
                    val_outputs_last_timestep = val_outputs[:, -1, :]
                    val_loss = torch.sqrt(criterion(val_outputs_last_timestep, Y_val))
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
            else:
                # No validation data for this epoch
                self.val_losses.append(None)

            if (epoch + 1) % 20 == 0 and self.verbose:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}"
                )
                if val_loader:
                    logger.info(f", Val Loss: {avg_val_loss:.4f}")

        logger.info("Training Complete")

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained model on the provided dataloader.

        Args:
            dataloader (DataLoader): The dataloader containing the data to make predictions on.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The first array
            is the model's predictions, and the second array is the actual target values from the dataloader.
        """

        all_predictions = []
        all_targets = []

        with torch.no_grad():  # Deactivate gradients for the following block
            for (
                X_batch,
                Y_batch,
                player_team_ids_batch,
                opponent_team_ids_batch,
                date_ids_batch,
                country_ids_batch,
            ) in dataloader:
                Y_batch = Y_batch.to(self.device)
                # Transfer to GPU if available
                X_batch = X_batch.to(self.device)
                player_team_ids_batch = player_team_ids_batch.to(self.device)
                opponent_team_ids_batch = opponent_team_ids_batch.to(self.device)
                date_ids_batch = date_ids_batch.to(self.device)
                country_ids_batch = country_ids_batch.to(self.device)

                # Forward pass
                outputs = self.forward(
                    X_batch,
                    player_team_ids_batch,
                    opponent_team_ids_batch,
                    date_ids_batch,
                    country_ids_batch,
                )

                # Since we're interested in the last timestep of the output sequence
                outputs_last_timestep = outputs[:, -1, :]

                # Store predictions
                all_predictions.append(outputs_last_timestep)
                all_targets.append(Y_batch)

        # Concatenate predictions from all batches
        predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        predictions = predictions * self.scale_factors
        all_targets = all_targets * self.scale_factors.view(1, -1)

        return predictions.cpu().numpy(), all_targets.cpu().numpy()

    def save_model(self, file_path):
        """
        Save the model's state dictionary to a file.

        Args:
        - file_path (str): The path to the file where the model should be saved.
        """
        torch.save(self.state_dict(), file_path)
        if self.verbose:
            logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """
        Load the model's state dictionary from a file.

        Args:
        - file_path (str): The path to the file from which to load the model.
        """
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        # Ensure the model parameters are on the correct device
        self.to(self.device)
        if self.verbose:
            logger.info(f"Model loaded from {file_path} to {self.device}")
