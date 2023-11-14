import boto3
import os
from moto import mock_s3
import pandas as pd


def is_close(a, b):
    if abs(a) > 0:
        return abs((a - b) / a) < 0.0001
    else:
        return abs((a - b)) < 0.0001


@mock_s3
def test_cli_run_with_model_key():
    # Arrange
    from nba.src.cli import parse_args, run

    bucket_name = "mybucket"
    model_key = "model.pth"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)
    # Read the real state dict file
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "player_box_score_predictor_state_dict.pth",
    )
    with open(model_path, "rb") as f:
        model_state_dict = f.read()

    # Upload the state dict to the mocked S3
    s3.put_object(Bucket=bucket_name, Key=model_key, Body=model_state_dict)

    sample_data_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "sample_input_data.csv",
    )

    test_args = [
        "--model-key",
        f"s3://{bucket_name}/{model_key}",
        "--input-file",
        sample_data_file,
    ]

    # Act
    args = parse_args(test_args)
    predictions = run(args)

    # Assert
    assert predictions is not None


def test_cli_run_without_model_key():
    # Should default to local model path
    # Arrange
    from nba.src.cli import parse_args, run
    from nba.src.constants.box_score_target_features import BoxScoreTargetFeatures

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
    """
    print("test output")
    print(predictions)
    print("sample output")
    print(sample_output)
    for column in BoxScoreTargetFeatures.list():
        assert all(
            is_close(x, y)
            for x, y in zip(
                predictions["predictions"][column].tolist(),
                sample_output[column].tolist(),
            )
        )
    """
