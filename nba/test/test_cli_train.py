import boto3
import os
import io
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
    trained_model = run(args)

    # Assert
    assert trained_model is not None
    assert isinstance(trained_model, dict)
    assert "model" in trained_model
    assert isinstance(trained_model["model"], PlayerEmbeddingLSTM)
