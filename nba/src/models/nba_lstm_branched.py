import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .base_model import BaseModel

torch_target_scaler = {
    "PTS": 81.0,
    "REB": 31.0,
    "AST": 25.0,
    "STL": 10.0,
    "BLK": 12.0,
    "minutes_played": 96.0,
    "TO": 12.0,
    "PF": 15.0,
    "PLUS_MINUS": 57.0,
}


# Configure logging
logger = logging.getLogger(__name__)


class PlayerEmbeddingLSTM(nn.Module, BaseModel):
    def __init__(
        self,
        input_size,
        max_hidden_size=32,
        dropout=0.2,
        output_size=1,
        max_num_players=2500,
        player_embedding_dim=16,
        clip_value=2,
        lstm_hidden_dim=32,
        lstm_layers=1,
        branch_lstm_hidden_dim=16,
        torch_target_scaler=[81, 31, 25, 10, 12],
        verbose=False,
    ):
        super(PlayerEmbeddingLSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout)
        self.player_embedding = nn.Embedding(max_num_players, player_embedding_dim)
        self.scale_factors = torch.tensor(torch_target_scaler, dtype=torch.float32).to(
            self.device
        )
        self.verbose = verbose
        self.train_losses = []
        self.val_losses = []
        self.clip_value = clip_value
        self.shared_base = nn.Linear(input_size + player_embedding_dim, lstm_hidden_dim)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.shared_lstm = nn.LSTM(
            lstm_hidden_dim,
            lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False,
        )

        # Separate LSTM and FC layers for each output
        self.lstm_pts = nn.LSTM(
            lstm_hidden_dim, branch_lstm_hidden_dim, batch_first=True
        )
        self.lstm_reb = nn.LSTM(
            lstm_hidden_dim, branch_lstm_hidden_dim, batch_first=True
        )
        self.lstm_ast = nn.LSTM(
            lstm_hidden_dim, branch_lstm_hidden_dim, batch_first=True
        )
        self.lstm_stl = nn.LSTM(
            lstm_hidden_dim, branch_lstm_hidden_dim, batch_first=True
        )
        self.lstm_blk = nn.LSTM(
            lstm_hidden_dim, branch_lstm_hidden_dim, batch_first=True
        )

        self.fc_mid_pts = nn.Linear(branch_lstm_hidden_dim, max_hidden_size)  # Points
        self.fc_mid_reb = nn.Linear(branch_lstm_hidden_dim, max_hidden_size)  # Rebounds
        self.fc_mid_ast = nn.Linear(branch_lstm_hidden_dim, max_hidden_size)  # Assists
        self.fc_mid_stl = nn.Linear(branch_lstm_hidden_dim, max_hidden_size)  # Steals
        self.fc_mid_blk = nn.Linear(branch_lstm_hidden_dim, max_hidden_size)  # Blocks

        self.fc_pts = nn.Linear(max_hidden_size, output_size)  # Points
        self.fc_reb = nn.Linear(max_hidden_size, output_size)  # Rebounds
        self.fc_ast = nn.Linear(max_hidden_size, output_size)  # Assists
        self.fc_stl = nn.Linear(max_hidden_size, output_size)  # Steals
        self.fc_blk = nn.Linear(max_hidden_size, output_size)  # Blocks

    def forward(self, x, player_ids):
        # Embedding and shared LSTM layer
        player_embed = self.player_embedding(player_ids)
        x = torch.cat([x, player_embed], dim=2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.shared_base(x)
        self.shared_lstm.flatten_parameters()
        shared_lstm_out, _ = self.shared_lstm(x)

        # Flatten parameters for the individual LSTMs
        self.lstm_pts.flatten_parameters()
        self.lstm_reb.flatten_parameters()
        self.lstm_ast.flatten_parameters()
        self.lstm_stl.flatten_parameters()
        self.lstm_blk.flatten_parameters()
        # Branch LSTM and FC layers for each output
        pts_lstm_out, _ = self.lstm_pts(shared_lstm_out)
        reb_lstm_out, _ = self.lstm_reb(shared_lstm_out)
        ast_lstm_out, _ = self.lstm_ast(shared_lstm_out)
        stl_lstm_out, _ = self.lstm_stl(shared_lstm_out)
        blk_lstm_out, _ = self.lstm_blk(shared_lstm_out)

        pts = torch.relu(self.dropout(self.fc_mid_pts(pts_lstm_out)))
        reb = torch.relu(self.dropout(self.fc_mid_reb(reb_lstm_out)))
        ast = torch.relu(self.dropout(self.fc_mid_ast(ast_lstm_out)))
        stl = torch.relu(self.dropout(self.fc_mid_stl(stl_lstm_out)))
        blk = torch.relu(self.dropout(self.fc_mid_blk(blk_lstm_out)))

        pts = torch.relu(self.fc_pts(pts))
        reb = torch.relu(self.fc_reb(reb))
        ast = torch.relu(self.fc_ast(ast))
        stl = torch.relu(self.fc_stl(stl))
        blk = torch.relu(self.fc_blk(blk))

        return pts, reb, ast, stl, blk

    def train(self, train_loader, val_loader=None, learning_rate=0.0001, epochs=10):
        self.to(self.device)
        criterion = nn.MSELoss()
        # optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=5
        )

        epochs = range(epochs) if self.verbose else tqdm(range(epochs))

        for epoch in epochs:
            total_train_loss = 0
            for X_batch, Y_batch, player_ids_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                player_ids_batch = player_ids_batch.to(self.device)

                # Forward pass
                (
                    outputs_pts,
                    outputs_reb,
                    outputs_ast,
                    outputs_stl,
                    outputs_blk,
                ) = self.forward(X_batch, player_ids_batch)

                # Separate losses for each output
                loss_pts = torch.sqrt(criterion(outputs_pts[:, -1, 0], Y_batch[:, 0]))
                loss_reb = torch.sqrt(criterion(outputs_reb[:, -1, 1], Y_batch[:, 1]))
                loss_ast = torch.sqrt(criterion(outputs_ast[:, -1, 2], Y_batch[:, 2]))
                loss_stl = torch.sqrt(criterion(outputs_stl[:, -1, 3], Y_batch[:, 3]))
                loss_blk = torch.sqrt(criterion(outputs_blk[:, -1, 4], Y_batch[:, 4]))
                # Combine losses
                total_loss = loss_pts + loss_reb + loss_ast + loss_stl + loss_blk

                optimizer.zero_grad()
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_value)
                optimizer.step()

                total_train_loss += total_loss.item()  # Accumulate the batch loss

            avg_train_loss = total_train_loss / len(
                train_loader
            )  # Compute average training loss for the epoch
            self.train_losses.append(avg_train_loss)  # Append the average training loss

            # Compute validation loss if validation data is provided
            if val_loader:
                total_val_loss = 0
                for X_val, Y_val, player_ids_val in val_loader:
                    X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
                    player_ids_val = player_ids_val.to(self.device)

                    # Forward pass
                    (
                        outputs_pts,
                        outputs_reb,
                        outputs_ast,
                        outputs_stl,
                        outputs_blk,
                    ) = self.forward(X_val, player_ids_val)

                    # Separate losses for each output
                    loss_pts = torch.sqrt(criterion(outputs_pts[:, -1, 0], Y_val[:, 0]))
                    loss_reb = torch.sqrt(criterion(outputs_reb[:, -1, 1], Y_val[:, 1]))
                    loss_ast = torch.sqrt(criterion(outputs_ast[:, -1, 2], Y_val[:, 2]))
                    loss_stl = torch.sqrt(criterion(outputs_stl[:, -1, 3], Y_val[:, 3]))
                    loss_blk = torch.sqrt(criterion(outputs_blk[:, -1, 4], Y_val[:, 4]))

                    # Combine losses
                    val_loss = loss_pts + loss_reb + loss_ast + loss_stl + loss_blk
                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                self.val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
            else:
                self.val_losses.append(None)  # No validation data for this epoch
                scheduler.step(avg_train_loss)

            if (epoch + 1) % 20 == 0 and self.verbose:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss.item():.4f}",
                    end="",
                )
                if val_loader:
                    logger.info(f", Val Loss: {avg_val_loss:.4f}")

    def predict(self, dataloader):
        self.eval()
        # Initialize lists to store predictions and targets for each stat
        (
            all_predictions_pts,
            all_predictions_reb,
            all_predictions_ast,
            all_predictions_stl,
            all_predictions_blk,
        ) = ([], [], [], [], [])
        all_targets = []

        with torch.no_grad():  # Deactivate gradients for the following block
            for X_batch, Y_batch, player_ids_batch in dataloader:
                Y_batch = Y_batch.to(self.device)
                # Transfer to GPU if available
                X_batch = X_batch.to(self.device)
                player_ids_batch = player_ids_batch.to(self.device)

                # Forward pass
                (
                    outputs_pts,
                    outputs_reb,
                    outputs_ast,
                    outputs_stl,
                    outputs_blk,
                ) = self.forward(X_batch, player_ids_batch)

                # Store predictions for each stat
                all_predictions_pts.append(outputs_pts[:, -1, 0])
                all_predictions_reb.append(outputs_reb[:, -1, 1])
                all_predictions_ast.append(outputs_ast[:, -1, 2])
                all_predictions_stl.append(outputs_stl[:, -1, 3])
                all_predictions_blk.append(outputs_blk[:, -1, 4])

                all_targets.append(Y_batch)

        # Concatenate predictions from all batches for each stat
        predictions_pts = torch.cat(all_predictions_pts, dim=0)
        predictions_reb = torch.cat(all_predictions_reb, dim=0)
        predictions_ast = torch.cat(all_predictions_ast, dim=0)
        predictions_stl = torch.cat(all_predictions_stl, dim=0)
        predictions_blk = torch.cat(all_predictions_blk, dim=0)

        # Concatenate all predictions into a single tensor
        predictions = torch.stack(
            [
                predictions_pts,
                predictions_reb,
                predictions_ast,
                predictions_stl,
                predictions_blk,
            ],
            dim=-1,
        ).squeeze()

        # Concatenate all targets
        all_targets = torch.cat(all_targets, dim=0)

        # Apply scale factors
        predictions = predictions * self.scale_factors
        all_targets = all_targets * self.scale_factors.view(1, -1)

        return predictions.cpu().numpy(), all_targets.cpu().numpy()

    def get_player_embeddings(self, player_id):
        player_id = torch.tensor(player_id).to(self.device)
        with torch.no_grad():
            return self.player_embedding(player_id)

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

    def eval(self):
        nn.Module.train(self, mode=False)

