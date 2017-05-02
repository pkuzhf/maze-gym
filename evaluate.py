import numpy as np
import config, utils


def evaluate(agent_net):

    n = config.Map.Height
    m = config.Map.Width

    mazemap = []
    for i in range(n):
        mazemap.append([])
        for j in range(m):
            mazemap[i].append(np.zeros(4))
            utils.setCellValue(mazemap, i, j, np.random.binomial(1, config.Map.WallDense))

    cell_value_memory = {}
    open(config.StrongMazeEnv.EvaluateFile, 'w')

    for distance in range(1, n + m):
        sum_score = 0
        for sx in range(n):
            for sy in range(m):
                if utils.getCellValue(mazemap, sx, sy) == config.Cell.Wall:
                    continue
                utils.setCellValue(mazemap, sx, sy, config.Cell.Source)
                score = 0
                count = 0
                output = ''
                for tx in range(n):
                    for ty in range(m):
                        if utils.getCellValue(mazemap, tx, ty) == config.Cell.Empty and utils.getDistance(sx, sy, tx, ty) <= distance:
                            count += 1
                            utils.setCellValue(mazemap, tx, ty, config.Cell.Target)
                            memory_id = str(sx) + '_' + str(sy) + '_' + str(tx) + '_' + str(ty)
                            if memory_id in cell_value_memory:
                                dir_id = cell_value_memory[memory_id]
                            else:
                                dir_id = np.array(agent_net.predict(np.array([[mazemap]]))).argmax()
                                cell_value_memory[memory_id] = dir_id
                            output += utils.dir_symbols[dir_id]
                            utils.setCellValue(mazemap, tx, ty, config.Cell.Empty)
                            if utils.getDistance(sx, sy, tx, ty) > utils.getDistance(sx + utils.dirs[dir_id][0], sy + utils.dirs[dir_id][1], tx, ty):
                                score += 1
                sum_score += float(score) / count
                utils.setCellValue(mazemap, sx, sy, config.Cell.Empty)
        sum_score /= n * m
        print [distance, sum_score]
        f = open(config.StrongMazeEnv.EvaluateFile, 'a')
        f.write(str(distance) + '\t' + str(sum_score) + '\n')
        f.close()
