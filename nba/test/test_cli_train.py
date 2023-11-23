import boto3
import os
import io
import tempfile
import torch
import pandas as pd
from moto import mock_s3


@mock_s3
def test_cli_train_without_model_key():
    # Should default to local model path
    # Arrange
    from nba.src.cli import parse_args, run
    from nba.src.models.nba_lstm_branched import PlayerEmbeddingLSTM

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="model-artifacts")

    sample_input_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_train_data.csv",
    )
    test_args = ["train", "--input-file", sample_input_file]

    # Act
    args = parse_args(test_args)
    results = run(args)

    with tempfile.NamedTemporaryFile() as temp_file:
        # Download the file from S3 to the temporary file
        s3.download_file(results["s3_bucket"], results["s3_key"], temp_file.name)
        # Read the CSV file into a DataFrame
        trained_model_state_dict = torch.load(temp_file.name)

    # Assert
    assert results is not None
    assert isinstance(results, dict)
    assert trained_model_state_dict is not None
    assert isinstance(trained_model_state_dict, dict)
    mock_model = PlayerEmbeddingLSTM(
        input_size=28,
        output_size=5,
        max_hidden_size=16,
        dropout=0.4,
        max_num_players=2500,
        player_embedding_dim=32,
        clip_value=2,
        lstm_hidden_dim=16,
        lstm_layers=1,
        branch_lstm_hidden_dim=8,
        torch_target_scaler=[1,2,3,4,5],
        verbose=False,
    )
    assert mock_model.load_state_dict(trained_model_state_dict)
