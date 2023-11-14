from ..pipeline.pipeline_state import PipelineState


class NbaLstmPipelineState(PipelineState):
    def __init__(self, state_name=None):
        super().__init__(state_name)

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value

    def remove(self, key):
        self.data.pop(key, None)

    def __str__(self):
        return str(self.data.keys()) + '/n' + str(self.data)
    
    def __repr__(self):
        return str(self.data.keys()) + '/n' + str(self.data)
