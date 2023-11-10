# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file into the container at /usr/src/app
COPY ../model/requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory into the container at /usr/src/app/model
COPY ../model/ ./model

# Make port available to the world outside this container
EXPOSE 80

# Define environment variable for S3 bucket and key
ENV MODEL_S3_BUCKET=my-model-bucket
ENV MODEL_S3_KEY=my-model-key

# Run inference.py when the container launches
CMD ["python", "model/src/inference.py", "--model-bucket", "$MODEL_S3_BUCKET", "--model-key", "$MODEL_S3_KEY"]
