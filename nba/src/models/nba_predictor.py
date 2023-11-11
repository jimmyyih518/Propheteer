from .base_model import BaseModel


class NbaPredictor(BaseModel):
    def __init__(self, model_path, data_processor):
        super().__init__(model_path)
        self.data_processor = data_processor
        self.load_model()

    def load_model(self):
        # Download from s3 self.path
        pass

    def predict(self, data):
        processed_sequence_data = self.data_processor.process_data(data)
        # predictions = self.model.predict(processed_sequence_data)
        return 0  # Not Implemented Yet

    def train(self, data):
        processed_sequence_data = self.data_processor.process_data(data)
        # self.model.train(processed_sequence_data)

    def save_model(self):
        # Upload to s3
        pass

    def forward(self):
        pass
