# def main():
#     print("Hello World")
#     print("This program computes the averate of two exam ")

#     score1, score2 = map(int, input("Enter two scores seperated by a commd: ").split())
#     #score1 = float(score1)
#     #score2 = float(score2)

#     average = (score1 + score2) / 2.0

#     print("The average of the score is : ", average) 

class HelloWorld():
    def __init__(self):
        print("Hello World")
    def __del__(self):
        print("Good bye")
    def performAverage(self, var1, var2):
        average = (var1 + var2) / 2.0
        print("The average of the scores is : ", average)

def main():
    world = HelloWorld()
    score1, score2 = map(int, input("Enter two scores seperated by a commd: ").split())
    world.performAverage(score1, score2)
    
main()