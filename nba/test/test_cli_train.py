import boto3
import os
import io
import pandas as pd
from moto import mock_s3


def test_cli_train_without_model_key():
    # Should default to local model path
    # Arrange
    from nba.src.cli import parse_args, run
    from nba.src.models.player_box_score_predictor import PlayerBoxScoreLSTM

    sample_input_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor",
        "sample_input_data.csv",
    )
    sample_output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor",
        "sample_output_data.csv",
    )
    sample_output = pd.read_csv(sample_output_file)

    test_args = ["train", "--input-file", sample_input_file]

    # Act
    args = parse_args(test_args)
    trained_model = run(args)

    # Assert
    assert trained_model is not None
    assert isinstance(trained_model, dict)
    assert "model" in trained_model
    assert isinstance(trained_model["model"], PlayerBoxScoreLSTM)
