#!/bin/bash

# Stop the script if any command fails
set -e

# Define variables
ECR_REPOSITORY="propheteer-precompute-worker-ecr"
AWS_REGION="us-west-2"
IMAGE_TAG="latest"

# Get the account number associated with the current IAM credentials
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

if [ $? -ne 0 ]; then
    echo "Error: AWS CLI not configured properly."
    exit 1
fi

# Define the full image name
FULL_IMAGE_NAME="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}"

# Authenticate Docker to ECR registry
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build the Docker image
docker build -f container/Dockerfile -t ${FULL_IMAGE_NAME} .

# Tag the image
docker tag ${FULL_IMAGE_NAME} ${FULL_IMAGE_NAME}

# Push the image to ECR
docker push ${FULL_IMAGE_NAME}

echo "Docker image pushed to ECR: ${FULL_IMAGE_NAME}"
