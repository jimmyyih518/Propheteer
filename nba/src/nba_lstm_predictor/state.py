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
        string_repr = (
            "State Keys: "
            + str(self.data.keys())
            + "/n"
            + "State Data: "
            + str(self.data)
        )
        return string_repr

    def __repr__(self):
        return self.__str__()
