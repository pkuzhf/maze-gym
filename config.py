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
	EnvTrainEpsForB = 0.5
	EnvTrainEpsMin = 0.1
	EnvTrainHalfEpsStep = 5000
	EnvWarmup = 50
	EnvTargetModelUpdate = 500
	EnvLearningRate = 1e-3
	AgentTrainEpsForB = 0.5
	AgentTrainEpsMin = 0.1
	AgentTrainHalfEpsStep = 5000
	AgentTestEpsForB = 0.1
	AgentWarmup = 50
	AgentTargetModelUpdate = 1e-3
	AgentLearningRate = 1e-3
