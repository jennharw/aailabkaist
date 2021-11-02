from Maze import Maze
from Move import Move

class Robot:
    def __init__(self):
        pass
    def findPath(self, startX, startY, endX, endY):
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

        NUM_DIRECTIONS = 4
        WIDTH = 8
        HEIGHT = 8
        DIRECTION_OFFSETS = [[0,-1], [1,0], [0,1], [-1,0]]

        NOTVISIT = 0
        WALL = 1
        VISIT = 2

        markArray = [[0 for _ in range(8)] for _ in range(8)]
        stack = [] ## push Move

        isEmpty = False
        isFound = False

        ##미로찾기
        start = Move(startX, startY)
        start.direction = 0
        stack.append(start)

        print("찾기")

        while isEmpty == False and isFound == False:
            curpos = stack.pop()
            x = curpos.x
            y = curpos.y
            direction = curpos.direction

            while isFound == False  and direction < NUM_DIRECTIONS :
                newX = x  + DIRECTION_OFFSETS[direction][0]
                newY = y + DIRECTION_OFFSETS[direction][1]
                
                #print(newX, newY)
                #print(M.maze[newY][newX] == NOTVISIT)
                #print(markArray[newY][newX] == NOTVISIT)

                if(newX >= 0 and newX < WIDTH and newY >= 0 and newY < HEIGHT and M.maze[newY][newX] == NOTVISIT and markArray[newY][newX] == NOTVISIT):
                    newPosition = Move(newX, newY)
                    newPosition.direction = direction + 1
                    stack.append(newPosition)
                    markArray[y][x] = VISIT

                    x = newX
                    y = newY
                    print(x, y)
                    direction = 0

                    if (newX == endX and newY == endY):
                        isFound = True
                        stack.append(newPosition)
                        markArray[y][x] = VISIT
                else:
                    direction += 1
R = Robot()
R.findPath(0,0,7,7)



# #1
# curpos = stack.pop()
# x = curpos.x
# y = curpos.y
# direction = curpos.direction

# while direction < NUM_DIRECTIONS :
#     newX = x  + DIRECTION_OFFSETS[direction][0]
#     newY = y + DIRECTION_OFFSETS[direction][1]
    
#     #print(newX, newY)
#     #print(M.maze[newY][newX] == NOTVISIT)
#     #print(markArray[newY][newX] == NOTVISIT)

#     if(newX >= 0 and newX < WIDTH and newY >= 0 and newY < HEIGHT and M.maze[newY][newX] == NOTVISIT and markArray[newY][newX] == NOTVISIT):
#         newPosition = Move(newX, newY)
#         newPosition.direction = direction + 1
#         stack.append(newPosition)
#         markArray[y][x] = VISIT

#         x = newX
#         y = newY
#         print(x, y)
#         direction = 0
#     else:
#         #print("안돼")
#         direction += 1
# #print(direction)
# print(stack)
# curpos = stack.pop()
# x = curpos.x
# y = curpos.y
# direction = curpos.direction

# print(curpos.x, curpos.y, curpos.direction)
# while direction < NUM_DIRECTIONS :
#     newX = x  + DIRECTION_OFFSETS[direction][0]
#     newY = y + DIRECTION_OFFSETS[direction][1]
    
#     #print(newX, newY)
#     #print(M.maze[newY][newX] == NOTVISIT)
#     #print(markArray[newY][newX] == NOTVISIT)

#     if(newX >= 0 and newX < WIDTH and newY >= 0 and newY < HEIGHT and M.maze[newY][newX] == NOTVISIT and markArray[newY][newX] == NOTVISIT):
#         newPosition = Move(newX, newY)
#         newPosition.direction = direction + 1
#         stack.append(newPosition)
#         markArray[y][x] = VISIT

#         x = newX
#         y = newY
#         print(x, y)
#         direction = 0
#     else:
#         print("안돼")
#         direction += 1

# curpos = stack.pop()
# x = curpos.x
# y = curpos.y
# direction = curpos.direction

# print(curpos.x, curpos.y, curpos.direction)
# while direction < NUM_DIRECTIONS :
#     newX = x  + DIRECTION_OFFSETS[direction][0]
#     newY = y + DIRECTION_OFFSETS[direction][1]
    
#     #print(newX, newY)
#     #print(M.maze[newY][newX] == NOTVISIT)
#     #print(markArray[newY][newX] == NOTVISIT)

#     if(newX >= 0 and newX < WIDTH and newY >= 0 and newY < HEIGHT and M.maze[newY][newX] == NOTVISIT and markArray[newY][newX] == NOTVISIT):
#         newPosition = Move(newX, newY)
#         newPosition.direction = direction + 1
#         stack.append(newPosition)
#         markArray[y][x] = VISIT

#         x = newX
#         y = newY
#         print(x, y)
#         direction = 0
#     else:
#        # print("안돼")
#         direction += 1

# endX = 7
# endY = 7

