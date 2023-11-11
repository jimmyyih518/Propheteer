from .base_model import BaseModel


class NbaPredictor(BaseModel):
    def __init__(self, model_path, data_processor):
        super().__init__(model_path)
        self.data_processor = data_processor
        if not self.model:
            self.load_model()

    def load_model(self):
        if self.model_path.startswith("s3"):
            self.load_model_from_s3()
        else:
            self.load_model_from_local()

    def load_model_from_s3(self):
        pass

    def load_model_from_local(self):
        pass

    def predict(self, data):
        processed_sequence_data = self.data_processor.process_data(data)
        # predictions = self.model.predict(processed_sequence_data)
        return 0  # Not Implemented Yet
