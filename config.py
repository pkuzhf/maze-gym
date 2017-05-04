import numpy as np

class Map:
	Height = 8
	Width = 8
	WallDense = 0.

class Cell:
	Empty = np.asarray([1,0,0,0])
	Wall = np.asarray([0,1,0,0])
	Source = np.asarray([0,0,1,0])
	Target = np.asarray([0,0,0,1])
	CellSize = 4

class StrongMazeEnv:
	ScoreLevel = 0.8
	EvaluateFile = '/tmp/evaluate.txt'

class Generator:
	RewardSampleN = 5
	ExploreRate = 0.05

class Game:
	MaxGameStep = 500
	Seed = 123456