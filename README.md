# Propheteer
A model for predicting NBA Player boxscore stats, deployed on AWS (WIP)

Test docker image build locally:

`./scripts/build.sh`

`docker run -it <image_name> bash`


Deploy docker image to ECR:
`./scripts/deploy.sh`


Running model in local python environment:
`python3 -m nba.src.cli --input-file nba/src/artifacts/nba_lstm_predictor_v2/sample_input_data.csv --local-dir ./tmp/`

Running model in local docker environment with build:
`./scripts/run.sh /path/to/input/file.csv --input-file value1 --local-dir value2`