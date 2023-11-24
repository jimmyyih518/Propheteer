import boto3
import os
import io
import tempfile
import torch
import pandas as pd
from moto import mock_s3
import json
from pathlib import Path

project_root = Path(__file__).parents[1]


@mock_s3
def test_cli_train_without_model_key():
    # Should default to local model path
    # Arrange
    from nba.src.cli import parse_args, run
    from nba.src.models.nba_lstm_branched import PlayerEmbeddingLSTM

    with open(
        f"{project_root}/src/artifacts/nba_lstm_predictor_v2/lstm_model_branched_config.json"
    ) as file:
        model_params = json.load(file)

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
    mock_model = PlayerEmbeddingLSTM(**model_params)
    assert mock_model.load_state_dict(trained_model_state_dict)
