import argparse
from .models.nba_predictor import NbaPredictor


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="NBA Predictor CLI Tool")
    parser.add_argument(
        "--param1", type=str, required=True, help="Description for param1"
    )
    parser.add_argument(
        "--param2", type=int, required=True, help="Description for param2"
    )
    return parser.parse_args(args)


def run(args):
    predictor = NbaPredictor(model_path=args.param1)
    predictions = predictor.predict(args.param2)
    print("Predictions made")
    return predictions


if __name__ == "__main__":
    args = parse_args()
    run(args)
