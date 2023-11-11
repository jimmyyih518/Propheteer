import argparse
from .models.nba_predictor import NbaPredictor
from .data_processors.sequence_processor import NbaSequenceDataProcessor


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="NBA Predictor CLI Tool")
    parser.add_argument(
        "--model-key", type=str, required=True, help="S3 key for model artifact"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="S3 key for input file"
    )
    return parser.parse_args(args)


def run(args):
    box_score_sequence_processor = NbaSequenceDataProcessor()
    predictor = NbaPredictor(
        model_path=args.model_key, data_processor=box_score_sequence_processor
    )
    predictions = predictor.predict(args.input_file)
    print("Predictions made")
    return predictions


if __name__ == "__main__":
    args = parse_args()
    run(args)
