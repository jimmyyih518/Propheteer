def test_cli_run():
    from nba.src.cli import parse_args, run

    # Arrange
    test_args = ["--model-key", "value1", "--input-file", "value2"]

    # Act
    args = parse_args(test_args)

    predictions = run(args)

    # Assert
    assert predictions is not None
