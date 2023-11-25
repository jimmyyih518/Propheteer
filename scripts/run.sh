#!/bin/bash

# Stop the script if any command fails
set -e

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input-file> [cli-args...]"
    exit 1
fi

DOCKER_LOCAL_DIR="/usr/src/app/runfiles/"

# Extract the full path of the input file
INPUT_FILE=$(realpath $1)

# Extract the full directory path of the input file
INPUT_DIR_PATH=$(dirname "$INPUT_FILE")

# Extract only the last part of the directory path (folder name)
INPUT_DIR=$(basename "$INPUT_DIR_PATH")

DOCKER_INPUT_FILENAME="$DOCKER_LOCAL_DIR$(basename $INPUT_FILE)"

echo "Input File:"
echo $INPUT_FILE
echo "Input Directory:"
echo $INPUT_DIR
echo "Processed Docker input file:"
echo $DOCKER_INPUT_FILENAME

# The rest of the arguments are for the Docker container CLI
CLI_ARGS=${@:2}

DOCKER_IMAGE="nba_lstm_predictor_local"

# Build the Docker image
docker build -f container/Dockerfile -t ${DOCKER_IMAGE} .

# Tag the image
docker tag ${DOCKER_IMAGE} ${DOCKER_IMAGE}

# Run the Docker container with volume mapping and CLI arguments
docker run -v $INPUT_DIR_PATH:/usr/src/app/runfiles \
    $DOCKER_IMAGE python nba/src/cli.py \
    --input-file $DOCKER_INPUT_FILENAME \
    --local-dir ./$INPUT_DIR/ \
    $CLI_ARGS

