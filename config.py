class Game:
    Seed = 123456
    MaxGameStep = 500

class Map:
    Height = 2
    Width = 2
    WallDense = 0.

class Generator:
    RolloutSampleN = 10
    ExploreRate = 0.05

class StrongMazeEnv:
    ScoreLevel = 0.8
    EvaluateFile = '/tmp/evaluate.txt'

