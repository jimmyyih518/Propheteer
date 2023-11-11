from .base_model import BaseModel


class NbaPredictor(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.load_model()

    def load_model(self):
        # Download from s3 self.path
        pass

    def predict(self, data):
        return 0 # Not Implemented Yet
