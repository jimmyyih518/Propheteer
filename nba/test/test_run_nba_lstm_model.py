import os
import pandas as pd


def test_cli_run_without_model_key():
    # Should default to local model path
    # Arrange
    from nba.src.run_nba_lstm_model import parse_args, run

    sample_input_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "sample_input_data.csv",
    )
    sample_output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "sample_output_data.csv",
    )
    sample_output = pd.read_csv(sample_output_file)

    test_args = ["--input-file", sample_input_file]

    # Act
    args = parse_args(test_args)
    predictions = run(args)

    # Assert
    assert predictions is not None
