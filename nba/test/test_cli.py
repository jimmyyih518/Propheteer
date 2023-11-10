def test_cli_run():
    from nba.src.cli import parse_args, run

    # Arrange
    test_args = ["--param1", "value1", "--param2", "123"]

    # Act
    args = parse_args(test_args)

    predictions = run(args)

    # Assert
    assert predictions is not None
