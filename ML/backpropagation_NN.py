#%%
import numpy as np
import time

#함수
def _m(A, B):
    return np.matmul(A, B)

def _t(x):
    return np.transpose(x)

#Sigmoid
class Sigmoid:
    def __init__(self):
        self.last_o = 1
    def __call__(self, x):
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o
    def grad(self):
        return self.last_o * (1 - self.last_o)

#Mean Squared Error
class MSE:
    def __init__(self):
        self.dh = 1 # Loss 의 derivate gradient
        self.diff = 1
    def __call__(self, h, y):
        self.diff = h - y
        return 1/2 * np.mean(np.square(h - y))
    def grad(self):
        return self.diff

#Neuron - layer
class Neuron:
    def __init__(self, W, b, a):
        self.W = W
        self.b = b
        self.a = a()

        #gradient
        self.dh = np.zeros_like(_t(self.W))  #chain rule
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        #last_x, last_h
        self.last_x = np.zeros_like((self.W.shape[0])) 
        self.last_h = np.zeros_like((self.W.shape[1]))

    def __call__(self, x):

        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b
        return self.a(self.last_h)
    
    def grad(self):
        return self.W * self.a.grad()
 
    def grad_W(self, dh): #dW
        grad = np.ones_like(self.W)

        for j in range(grad.shape[1]):
            grad[:, j] = self.last_x * self.a.grad()[j] * dh[j]
        return grad

    def grad_b(self, dh): #db
        return dh * self.a.grad()

#%%
class DNN:
    def __init__(self, hidden_depth, num_neuron, num_input, num_output, activation=Sigmoid):
        ##초기화 
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i,o)), np.zeros((o,))

        self.sequence = list()
        #first layer
        W, b = init_var(num_input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        #hidden layer
        for _ in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        #output layer
        W, b = init_var(num_neuron, num_output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x
    
    def calc_gradient(self, loss_obj): #dh _t(W)
        #backpropagation  - sequence 에 loss 추가해야
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)

        for i in range(len(self.sequence)-1, 0, -1): #[5,4,3,2,1]
            layer = self.sequence[i] #loss, 현 layer
            layer_0 = self.sequence[i - 1] 
            #dh - Dynamic Programming
            ##layer_0.grad() - fn / fn-1 = Wt
            ##layer.dh #loss 의 grad
            layer_0.dh = _m(layer_0.grad(), layer.dh) #grad() W, dh Wt A*Wt , A*W
            #dW
            layer_0.dW = layer_0.grad_W(layer.dh)
            #db
            layer_0.db = layer_0.grad_b(layer.dh)
        
        self.sequence.remove(loss_obj)



#경사하강법 Gradient Descent
def gradient_descent(network, x, y, loss_obj, alpha = 0.01):
    loss = loss_obj(network(x), y)
    network.calc_gradient(loss_obj)
    for layer in network.sequence:        
        layer.W += -alpha * layer.dW 
        layer.b += -alpha * layer.db 
    return loss

#training
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))
t = time.time()

dnn = DNN(hidden_depth=5, num_neuron=32, num_input = 10, num_output=2, activation=Sigmoid)
loss_obj = MSE()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha = 0.01)
    print(f'training accuracy : {loss}  - epoch : {epoch}')



