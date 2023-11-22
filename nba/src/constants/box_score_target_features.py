from enum import Enum


class BoxScoreTargetFeatures(Enum):
    PTS = "PTS"
    REB = "REB"
    AST = "AST"
    STL = "STL"
    BLK = "BLK"

    @classmethod
    def list(cls):
        return [member.value for member in cls]


class BoxScoreTargetFeaturesV2(Enum):
    PTS='PTS'
    REB='REB'
    AST='AST'
    STL='STL'
    BLK='BLK'

    @classmethod
    def list(cls):
        return [member.value for member in cls]