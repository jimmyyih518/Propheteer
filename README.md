# Propheteer
A model for predicting NBA Player boxscore stats, deployed on AWS (WIP)

Test docker image build locally:

`./scripts/build.sh`

`docker run -it <image_name> bash`


Deploy docker image to ECR (assumes AWS CDK deployment has completed):
`./scripts/deploy.sh`


## Local train and test
- Ensure the `runfiles` folder is in the root directory
- Any input files will be read in (or mounted onto docker) from the `runfiles` directory
- Any output files will be written to the `runfiles` directory

#### Running model inference in local python environment:
`python3 -m nba.src.cli --input-file runfiles/sample_input_data.csv --local-dir ./runfiles/`

#### Running model inference in local docker environment with build:
`./scripts/run.sh runfiles/sample_input_data.csv predict --local-dir ./runfiles/`

#### Running model training in local docker environment with build:
`./scripts/run.sh runfiles/sample_train_data.csv train --local-dir ./runfiles/`