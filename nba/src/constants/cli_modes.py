from enum import Enum


class CliRunModes(Enum):
    train = "train"
    predict = "predict"

    @classmethod
    def list(cls):
        return [member.value for member in cls]
