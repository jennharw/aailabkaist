class Maze:
    def __init__(self, size):
        self.size = size 
        self.maze = [[0 for _ in range(size)] for _ in range(size)]

M = Maze(8)
M.maze[0][0] = 0
M.maze[0][1] = 1
M.maze[0][2] = 1
M.maze[0][3] = 1
M.maze[0][4] = 0
M.maze[0][5] = 1
M.maze[0][6] = 1
M.maze[0][7] = 1

M.maze[1][0] = 0
M.maze[1][1] = 0
M.maze[1][2] = 0
M.maze[1][3] = 1
M.maze[1][4] = 0
M.maze[1][5] = 0
M.maze[1][6] = 0
M.maze[1][7] = 0

M.maze[2][0] = 1
M.maze[2][1] = 1
M.maze[2][2] = 0
M.maze[2][3] = 0
M.maze[2][4] = 0
M.maze[2][5] = 1
M.maze[2][6] = 0
M.maze[2][7] = 1

M.maze[3][0] = 1
M.maze[3][1] = 1
M.maze[3][2] = 0
M.maze[3][3] = 1
M.maze[3][4] = 1
M.maze[3][5] = 1
M.maze[3][6] = 0
M.maze[3][7] = 1

M.maze[4][0] = 1
M.maze[4][1] = 0
M.maze[4][2] = 0
M.maze[4][3] = 1
M.maze[4][4] = 0
M.maze[4][5] = 0
M.maze[4][6] = 0
M.maze[4][7] = 0

M.maze[5][0] = 0
M.maze[5][1] = 1
M.maze[5][2] = 1
M.maze[5][3] = 1
M.maze[5][4] = 0
M.maze[5][5] = 1
M.maze[5][6] = 1
M.maze[5][7] = 1

M.maze[6][0] = 1
M.maze[6][1] = 0
M.maze[6][2] = 1
M.maze[6][3] = 1
M.maze[6][4] = 0
M.maze[6][5] = 0
M.maze[6][6] = 0
M.maze[6][7] = 0

M.maze[6][0] = 0
M.maze[7][1] = 1
M.maze[7][2] = 1
M.maze[7][3] = 0
M.maze[7][4] = 1
M.maze[7][5] = 1
M.maze[7][6] = 1
M.maze[7][7] = 0

print(M.maze)