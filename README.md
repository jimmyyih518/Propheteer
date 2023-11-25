# Propheteer: NBA Fantasy Player Predictions
A model for predicting NBA Player boxscore stats, deployed on AWS (WIP)

Test docker image build locally:

`./scripts/build.sh`

`docker run -it <image_name> bash`


Deploy docker image to ECR (assumes AWS CDK deployment has completed, see [cdk README](https://github.com/jimmyyih518/Propheteer/blob/main/cdk/README.md)):

`./scripts/deploy.sh`


## Local train and test
- Ensure the `runfiles` folder is in the root directory
- Any input files will be read in (or mounted onto docker) from the `runfiles` directory
- Any output files will be written to the `runfiles` directory


#### (Recommended) Running model in local docker environment with build:
- Replace `sample_input_data.csv` and `sample_train_data.csv` with any compatible dataset

Inference Mode:

- Prepare a csv file for inferene input, formatted like [sample input file](https://github.com/jimmyyih518/Propheteer/blob/main/runfiles/sample_input_data.csv)
- `./scripts/run.sh runfiles/sample_input_data.csv predict --output-file runfiles/mypredictions`
- An output `.csv` predictions table will be saved into the `runfiles` directory

Train Mode:

- Prepare a csv file for training input, formatted like [sample training file](https://github.com/jimmyyih518/Propheteer/blob/main/runfiles/sample_train_data.csv)
- `./scripts/run.sh runfiles/sample_train_data.csv train --epochs 10 --learning-rate 0.0001 --seq-batch-size 32 --output-file runfiles/trained_model`
- An output `.pth` state dict of the trained model will be saved into the `runfiles` directory

#### Running model in local python environment
Inference Mode:

`python3 -m nba.src.cli predict --input-file runfiles/sample_input_data.csv --local-dir ./runfiles/`

Train Mode:
`python3 -m nba.src.cli train --input-file runfiles/sample_input_data.csv --local-dir ./runfiles/`
