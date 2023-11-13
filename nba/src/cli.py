import os
import argparse
import logging
from nba.src.models.nba_predictor import NbaPredictor
from nba.src.data_processors.sequence_processor import NbaSequenceDataProcessor
from nba.src.constants.cli_modes import CliRunModes


dir_path = os.path.dirname(os.path.realpath(__file__))
default_local_model_path = os.path.join(
    dir_path, "artifacts/player_box_score_predictor_state_dict.pth"
)
default_input_scaler_path = os.path.join(dir_path, "artifacts/lstm_input_scaler.pkl")
default_team_encoder_path = os.path.join(dir_path, "artifacts/lstm_team_encoder.pkl")
default_country_encoder_path = os.path.join(
    dir_path, "artifacts/lstm_country_encoder.pkl"
)

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="NBA Predictor CLI Tool")
    parser.add_argument(
        "mode",
        type=str,
        choices=CliRunModes.list(),
        default=CliRunModes.predict,
        nargs="?",
        help="Operation mode, 'train' or 'predict'. Defaults to 'predict'.",
    )
    parser.add_argument(
        "--model-key", type=str, required=False, help="S3 key for model artifact"
    )
    parser.add_argument(
        "--input-scaler-key", type=str, required=False, help="S3 key for model artifact"
    )
    parser.add_argument(
        "--team-encoder-key", type=str, required=False, help="S3 key for model artifact"
    )
    parser.add_argument(
        "--country-encoder-key",
        type=str,
        required=False,
        help="S3 key for model artifact",
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="S3 key for input file"
    )
    return parser.parse_args(args)


def run(args):
    input_scaler_path = (
        args.input_scaler_key if args.input_scaler_key else default_input_scaler_path
    )
    team_encoder_path = (
        args.team_encoder_key if args.team_encoder_key else default_team_encoder_path
    )
    country_encoder_path = (
        args.country_encoder_key
        if args.country_encoder_key
        else default_country_encoder_path
    )
    logger.info(f"Input Scaler path: {input_scaler_path}")
    logger.info(f"Team Encoder path: {team_encoder_path}")
    logger.info(f"Country Encoder path: {country_encoder_path}")
    box_score_sequence_processor = NbaSequenceDataProcessor(
        input_scaler_path, team_encoder_path, country_encoder_path
    )
    logger.info("Instantiated sequence data processor")

    model_path = args.model_key if args.model_key else default_local_model_path
    logger.info(f"Model path: {model_path}")
    model = NbaPredictor(
        model_path=model_path, data_processor=box_score_sequence_processor
    )
    if args.mode and args.mode == CliRunModes.train:
        logger.info("Training Model with input data")
        trained = model.train(args.input_file)
        logger.info("Model training completed")
        return trained
    else:
        logger.info("Generating predictions")
        predictions = model.predict(args.input_file)
        logger.info("Predictions made")
        return predictions


if __name__ == "__main__":
    args = parse_args()
    run(args)
