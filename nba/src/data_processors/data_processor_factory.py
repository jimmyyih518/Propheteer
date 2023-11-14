from .sequence_data_processor import SequenceDataProcessor
from .tabular_data_processor import TabularDataProcessor
from ..constants.model_types import ModelTypes


class DataProcessorFactory:
    @staticmethod
    def get_processor(model):
        if model.model_type == ModelTypes.SEQ:
            return SequenceDataProcessor()
        elif model.model_type == ModelTypes.TAB:
            return TabularDataProcessor()
        else:
            raise ValueError("Unknown model type")
