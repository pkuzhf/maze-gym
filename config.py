class Game:
    Seed = 123456
    MaxGameStep = 200

class Map:
    Height = 5
    Width = 5
    WallDense = 0.

class Generator:
    RolloutSampleN = 10
    ExploreRate = 0.05

class StrongMazeEnv:
    ScoreLevel = 0.8
    EvaluateFile = '/tmp/evaluate.txt'

class Training:
    RewardScale = 3

    EnvTrainEps = 1.0
    EnvTrainEps_Min = 0.1
    EnvTrainEps_HalfStep = 5000
    AgentTrainEps = 1.0
    AgentTrainEps_Min = 0.1
    AgentTrainEps_HalfStep = 5000

    EnvWarmup = 32
    AgentWarmup = 32

    EnvLearningRate = 1e-3
    AgentLearningRate = 1e-3

    EnvTargetModelUpdate = 1000
    AgentTargetModelUpdate = 1e-3

    AgentTestEps = 0.0
    EnvGenEps = 0.1
