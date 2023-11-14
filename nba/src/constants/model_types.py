from enum import Enum

class ModelTypes(Enum):
    SEQ = 'sequence'
    TAB = 'tabular'

    @classmethod
    def list(cls):
        return [member.value for member in cls]