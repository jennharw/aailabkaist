import numpy as np

#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Softmax Function
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)

#Network
class ShallowNN:
    def __init__(self, num_input, num_hidden, num_output):
        self.W_h = np.zeros((num_hidden, num_input), dtype=np.float32)
        self.b_h = np.zeros((num_hidden,), dtype = np.float32)
        self.W_o = np.zeros((num_output, num_hidden), dtype=np.float32)
        self.b_h = np.zeros((num_output,),dtype=np.float32)
    def __call__(self, x):
        h = sigmoid(np.matmul(self.W_h, x) + self.b_h)
        return softmax(np.matmul(self.W_o, h) + self.b_o)

#Dataset
#input
#label

#Model
model = ShallowNN(2, 128, 10)

#model parameter

#결과
output = model()
np.argmax(output)


