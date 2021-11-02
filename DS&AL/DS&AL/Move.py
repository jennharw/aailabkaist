#위치

class Move:
    x = 0
    y = 0
    def __init__(self, x , y):
        self.x = x
        self.y = y
        self.direction =0
move  = Move(0,0)
print(move.direction)
