import os
import argparse
import logging

from nba.src.pipeline import PipelineOrchestrator
from nba.src.nba_lstm_predictor_v2 import (
    state_machine,
    data_loader,
    feature_processor,
    sequence_processor,
    model,
    output_processor,
    output_writer,
)

from nba.src.constants.model_modes import ModelRunModes


dir_path = os.path.dirname(os.path.realpath(__file__))
default_local_model_path = os.path.join(
    dir_path, "artifacts/nba_lstm_predictor_v2/lstm_model_branched_state_dict.pth"
)
default_local_model_json = os.path.join(
    dir_path, "artifacts/nba_lstm_predictor_v2/lstm_model_branched_config.json"
)
default_input_scaler_path = os.path.join(
    dir_path, "artifacts/nba_lstm_predictor_v2/minmax_scaler.pkl"
)
default_player_encoder_path = os.path.join(
    dir_path, "artifacts/nba_lstm_predictor_v2/player_encoder_mapping.json"
)

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="NBA Predictor CLI Tool")
    parser.add_argument(
        "mode",
        type=str,
        choices=ModelRunModes.list(),
        default=ModelRunModes.predict.value,
        nargs="?",
        help="Operation mode, 'train' or 'predict'. Defaults to 'predict'.",
    )
    parser.add_argument(
        "--model-key", type=str, required=False, help="S3 key for model artifact"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=False,
        help="S3 key for model config json file",
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
    parser.add_argument(
        "--learning-rate", type=float, required=False, help="Learning Rate for Training"
    )
    parser.add_argument(
        "--epochs", type=int, required=False, help="Epochs for Training"
    )
    parser.add_argument(
        "--output-file", type=str, required=False, help="S3 key for output file"
    )
    return parser.parse_args(args)


def run(args):
    input_scaler_path = (
        args.input_scaler_key if args.input_scaler_key else default_input_scaler_path
    )
    player_encoder_path = (
        args.team_encoder_key if args.team_encoder_key else default_player_encoder_path
    )
    model_path = args.model_key if args.model_key else default_local_model_path
    model_config_path = (
        args.model_config if args.model_config else default_local_model_json
    )
    logger.info(f"Model path: {model_path}, invoking for mode {args.mode}")
    logger.info(f"Input Scaler path: {input_scaler_path}")
    logger.info(f"Player Encoder path: {player_encoder_path}")

    pipeline_state = state_machine()
    print('state id', pipeline_state.state_id)
    pipeline_data_loader = data_loader("data_loader")
    pipeline_feature_processor = feature_processor(
        component_name="feature_processor",
        input_scaler_path=input_scaler_path,
        player_encoder_path=player_encoder_path,
    )
    pipeline_sequence_processor = sequence_processor("sequence_processor")
    pipeline_model = model(
        "model",
        model_path=model_path,
        model_config_path=model_config_path,
        model_mode=args.mode,
    )
    pipeline_output_processor = output_processor("output_processor")
    pipeline_output_writer = output_writer("output_writer", args.output_file)
    pipeline = PipelineOrchestrator(
        name="nba_lstm_model",
        input_key=args.input_file,
        components=[
            pipeline_data_loader,
            pipeline_feature_processor,
            pipeline_sequence_processor,
            pipeline_model,
            pipeline_output_processor,
            pipeline_output_writer,
        ],
        return_last_output=True,
    )

    return pipeline.process(pipeline_state)


if __name__ == "__main__":
    args = parse_args()
    run(args)
