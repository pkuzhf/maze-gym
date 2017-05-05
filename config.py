class Game:
	Seed = 123456
	MaxGameStep = 500

class Map:
	Height = 8
	Width = 8
	WallDense = 0.

class Generator:
	RolloutSampleN = 5
	ExploreRate = 0.05

class StrongMazeEnv:
	ScoreLevel = 0.8
	EvaluateFile = '/tmp/evaluate.txt'

