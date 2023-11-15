from enum import Enum


class ModelRunModes(Enum):
    train = "train"
    predict = "predict"

    @classmethod
    def list(cls):
        return [member.value for member in cls]
