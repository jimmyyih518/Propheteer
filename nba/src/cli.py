import os
import argparse
import logging
from nba.src.models.nba_predictor import NbaPredictor
from nba.src.data_processors.sequence_processor import NbaSequenceDataProcessor


dir_path = os.path.dirname(os.path.realpath(__file__))
default_local_model_path = os.path.join(
    dir_path, "artifacts/player_box_score_predictor_state_dict.pth"
)

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="NBA Predictor CLI Tool")
    parser.add_argument(
        "--model-key", type=str, required=False, help="S3 key for model artifact"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="S3 key for input file"
    )
    return parser.parse_args(args)


def run(args):
    box_score_sequence_processor = NbaSequenceDataProcessor()
    model_path = args.model_key if args.model_key else default_local_model_path
    logger.info(f"Model path: {model_path}")
    predictor = NbaPredictor(
        model_path=model_path, data_processor=box_score_sequence_processor
    )
    logger.info("Generating predictions")
    predictions = predictor.predict(args.input_file)
    logger.info("Predictions made")
    return predictions


if __name__ == "__main__":
    args = parse_args()
    run(args)
