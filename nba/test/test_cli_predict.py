import boto3
import os
import io
import tempfile
import pandas as pd
from moto import mock_s3

from conftest import DEFAULT_TEST_DIR


def is_close(a, b):
    if abs(a) > 0:
        return abs((a - b) / a) < 0.0001
    else:
        return abs((a - b)) < 0.0001


@mock_s3
def test_cli_predict_with_s3_model_key():
    # Arrange
    from nba.src.cli import parse_args, run

    bucket_name = "mybucket"
    model_key = "model.pth"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)
    s3.create_bucket(Bucket="model-predictions")
    # Read the real state dict file
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "lstm_model_branched_state_dict.pth",
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
        "nba_lstm_predictor_v2",
        "sample_input_data.csv",
    )
    sample_output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_output_data.csv",
    )
    sample_output = pd.read_csv(sample_output_file)

    test_args = [
        "--model-key",
        f"s3://{bucket_name}/{model_key}",
        "--input-file",
        sample_data_file,
    ]

    # Act
    args = parse_args(test_args)
    results = run(args)

    with tempfile.NamedTemporaryFile() as temp_file:
        # Download the file from S3 to the temporary file
        s3.download_file(results["s3_bucket"], results["s3_key"], temp_file.name)
        # Read the CSV file into a DataFrame
        predictions = pd.read_csv(temp_file.name)

    # Assert
    assert results is not None
    assert isinstance(results, dict)
    assert isinstance(predictions, pd.DataFrame)


@mock_s3
def test_cli_predict_without_model_key():
    # Should default to local model path
    # Arrange
    from nba.src.cli import parse_args, run

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="model-predictions")

    sample_input_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_input_data.csv",
    )
    sample_output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_output_data.csv",
    )
    sample_output = pd.read_csv(sample_output_file)

    test_args = ["--input-file", sample_input_file]

    # Act
    args = parse_args(test_args)
    results = run(args)

    with tempfile.NamedTemporaryFile() as temp_file:
        # Download the file from S3 to the temporary file
        s3.download_file(results["s3_bucket"], results["s3_key"], temp_file.name)
        # Read the CSV file into a DataFrame
        predictions = pd.read_csv(temp_file.name)

    # Assert
    assert results is not None
    assert isinstance(results, dict)
    assert isinstance(predictions, pd.DataFrame)
    pd.testing.assert_frame_equal(
        predictions,
        sample_output,
        check_dtype=False,
        check_column_type=False,
    )


@mock_s3
def test_cli_predict_with_s3_input_key():
    # Arrange
    from nba.src.cli import parse_args, run

    bucket_name = "mybucket"
    file_key = "input.csv"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="model-predictions")
    s3.create_bucket(Bucket=bucket_name)
    # Read the real state dict file
    sample_data_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_input_data.csv",
    )
    sample_data = pd.read_csv(sample_data_file)
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)

    # Upload the state dict to the mocked S3
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())

    sample_output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_output_data.csv",
    )
    sample_output = pd.read_csv(sample_output_file)

    test_args = [
        "--input-file",
        f"s3://{bucket_name}/{file_key}",
    ]

    # Act
    args = parse_args(test_args)
    results = run(args)

    with tempfile.NamedTemporaryFile() as temp_file:
        # Download the file from S3 to the temporary file
        s3.download_file(results["s3_bucket"], results["s3_key"], temp_file.name)
        # Read the CSV file into a DataFrame
        predictions = pd.read_csv(temp_file.name)

    # Assert
    assert results is not None
    assert isinstance(results, dict)
    assert isinstance(predictions, pd.DataFrame)
    print("predictions", predictions)
    print("sample_output", sample_output)

    pd.testing.assert_frame_equal(
        predictions,
        sample_output,
        check_dtype=False,
        check_column_type=False,
    )


@mock_s3
def test_cli_predict_without_model_key_with_local_output():
    # Should default to local model path
    # Arrange
    from nba.src.cli import parse_args, run

    sample_input_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_input_data.csv",
    )
    sample_output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "artifacts",
        "nba_lstm_predictor_v2",
        "sample_output_data.csv",
    )
    sample_output = pd.read_csv(sample_output_file)

    test_args = ["--input-file", sample_input_file, "--local-dir", DEFAULT_TEST_DIR]

    # Act
    args = parse_args(test_args)
    results = run(args)
    predictions = pd.read_csv(results["output"])

    # Assert
    assert results is not None
    assert isinstance(results, dict)
    assert isinstance(predictions, pd.DataFrame)
    pd.testing.assert_frame_equal(
        predictions,
        sample_output,
        check_dtype=False,
        check_column_type=False,
    )

