# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"

# Copy the requirements file into the container at /usr/src/app
COPY ../nba/requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model directory into the container at /usr/src/app/model
COPY ../nba/ ./nba

# Make port available to the world outside this container
EXPOSE 80

# Run cli.py when the container launches (to be overriden when actuall running)
CMD ["python", "nba/src/cli.py", "--input-file-key", "sample-input-file.csv"]
