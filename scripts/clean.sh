#!/bin/bash

# Stop the script if any command fails
set -e

# List dangling Docker images
echo "Listing all dangling Docker images..."
docker images -f dangling=true

# Prune dangling Docker images
echo "Pruning dangling Docker images..."
docker image prune -f

# Completion statement
echo "Dangling images pruned successfully."
