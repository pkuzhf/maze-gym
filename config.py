class Game:
    Seed = 123456
    MaxGameStep = 200

class Map:
    Height = 8
    Width = 8
    WallDense = 0.

class Generator:
    RolloutSampleN = 10
    ExploreRate = 0.05

class StrongMazeEnv:
    ScoreLevel = 0.8
    EvaluateFile = '/tmp/evaluate.txt'

class Training:
	RewardScale = 3
	EnvTrainEpsForB = 0.1
	EnvWarmup = 50
	EnvTargetModelUpdate = 1e-3
	EnvLearningRate = 1e-3
	AgentTrainEpsForB = 0.1
	AgentTestEpsForB = 0.1
	AgentWarmup = 50
	AgentTargetModelUpdate = 1e-3
	AgentLearningRate = 1e-3