#!/bin/bash

# Stop the script if any command fails
set -e

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input-file> [cli-args...]"
    exit 1
fi

# Extract the full path of the input file
INPUT_FILE=$(realpath $1)

# Extract the directory of the input file
INPUT_DIR=$(dirname $INPUT_FILE)

# The rest of the arguments are for the Docker container CLI
CLI_ARGS=${@:2}

DOCKER_IMAGE="nba_lstm_predictor_local"

# Build the Docker image
docker build -f container/Dockerfile -t ${DOCKER_IMAGE} .

# Tag the image
docker tag ${DOCKER_IMAGE} ${DOCKER_IMAGE}

# Run the Docker container with volume mapping and CLI arguments
docker run -v $INPUT_DIR:/usr/src/app/data $DOCKER_IMAGE python nba/src/cli.py $CLI_ARGS

