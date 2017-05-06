class Game:
    Seed = 123456
    MaxGameStep = 500

class Map:
    Height = 4
    Width = 4
    WallDense = 0.

class Generator:
    RolloutSampleN = 16
    ExploreRate = 0.05

class StrongMazeEnv:
    ScoreLevel = 0.8
    EvaluateFile = '/tmp/evaluate.txt'

