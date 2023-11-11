import boto3
import os
from moto import mock_s3


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

    test_args = [
        "--model-key",
        f"s3://{bucket_name}/{model_key}",
        "--input-file",
        "value2",
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

    test_args = ["--input-file", "value2"]

    # Act
    args = parse_args(test_args)
    predictions = run(args)

    # Assert
    assert predictions is not None
