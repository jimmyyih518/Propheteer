from .base_model import BaseModel


class NbaPredictor(BaseModel):
    def __init__(self, model_path):
        super().__init__(model_path)

    def load_model(self):
        pass

    def predict(self, data):
        pass
