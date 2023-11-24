import os
import boto3
import tempfile
import torch
import pandas as pd
from typing import Any

from ..pipeline.pipeline_component import PipelineComponent
from ..constants.model_modes import ModelRunModes

S3_OUTPUT_BUCKET = {
    ModelRunModes.predict.value: "model-predictions",
    ModelRunModes.train.value: "model-artifacts",
}
OUTPUT_FILE_EXT = {
    ModelRunModes.predict.value: ".csv",
    ModelRunModes.train.value: ".pth",
}


class NbaLstmPredictorOutputWriter(PipelineComponent):
    def __init__(
        self,
        component_name: str,
        input_key: str = None,
        output_key: str = None,
        bucket: str = None,
    ):
        super().__init__(component_name, input_key)
        self.output_key = output_key
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def process(self, state: Any) -> None:
        data = state.get(self.input_key)
        self.run_mode = state.MODEL_RUNMODE
        self.output_key = state.state_id
        self.bucket = self.bucket if self.bucket else S3_OUTPUT_BUCKET[self.run_mode]
        self.file_key = self.output_key + OUTPUT_FILE_EXT[self.run_mode]
        response = self._write_output(data)
        state.set(self.component_name, response)

    def _write_output(self, data):
        if self.output_key.startswith("s3"):
            return self._write_output_to_s3(data)
        else:
            return self._write_output_to_local(data)

    def _write_output_to_s3(self, data):
        self.logger.info(f"Writing output to s3")
        if self.run_mode == ModelRunModes.predict.value:
            return self._write_prediction_to_s3(data)
        elif self.run_mode == ModelRunModes.train.value:
            return self._write_model_to_s3(data)

    def _write_prediction_to_s3(self, data):
        with tempfile.NamedTemporaryFile() as tmp:
            data["predictions"].to_csv(tmp.name, index=False)
            tmp.seek(0)  # Rewind the file to the beginning
            return {
                "s3_bucket": self.bucket,
                "s3_key": self.file_key,
                "s3_uri": f"s3://{self.bucket}/{self.file_key}",
                "response": self.s3.upload_fileobj(tmp, self.bucket, self.file_key),
            }

    def _write_model_to_s3(self, data):
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(data["model"].state_dict(), tmp.name)
            tmp.seek(0)  # Rewind the file to the beginning
            return {
                "s3_bucket": self.bucket,
                "s3_key": self.file_key,
                "s3_uri": f"s3://{self.bucket}/{self.file_key}",
                "response": self.s3.upload_fileobj(tmp, self.bucket, self.file_key),
            }

    def _write_output_to_local(self, data):
        local_dir = os.path.dirname(self.file_key)
        self.logger.info(f"Writing output to local dir {local_dir}")
        if not os.path.exists(local_dir):
            self.logger.info(f"Local dir {local_dir} doesn't exit, creating")
            os.makedirs(local_dir)

        if self.run_mode == ModelRunModes.predict.value:
            return self._write_prediction_to_local(data)
        elif self.run_mode == ModelRunModes.train.value:
            return self._write_model_to_local(data)

    def _write_prediction_to_local(self, data):
        data["predictions"].to_csv(self.file_key, index=False)
        return {"output": self.file_key}

    def _write_model_to_local(self, data):
        torch.save(data["model"].state_dict(), self.file_key)
        return {"output": self.file_key}
