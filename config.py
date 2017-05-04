class Map:
	Height = 8
	Width = 8
	WallDense = 0.

class Cell:
	Empty = 0
	Wall = 1
	Source = 2
	Target = 3

class StrongMazeEnv:
	ScoreLevel = 0.8
	EvaluateFile = '/tmp/evaluate.txt'

class Generator:
	RewardSampleN = 5
	ExploreRate = 0.05

class Game:
	MaxGameStep = 500
	Seed = 123