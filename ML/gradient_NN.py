import tensorflow as tf 
import numpy as np

#hyperparameter
epochs = 1000

#Nerual network
#input 2, hidden (sigmoid 128) ,output 10 (softmax)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(128, input_dim =2 , activation ='sigmoid')
        self.d2 = tf.keras.layers.Dense(10, activation = 'softamx') #input_dim 안적어도 됨 

    def __call__(self, x, training = None, mask = None):
        x = self.d1(x)
        return self.d2(x)


@tf.function
def train_step(mode, inputs, labels, loss_object, optimizer, train_loss, train_metric):
    with tf.GraidentTape() as tape:
        predictions =  model(inputs)
        loss = loss_object(labels, predictions)
    gradient = tape.gradient(loss, model.trainable_variables ) #loss scala 를 vector 로 미분
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    train_loss(loss)
    train_metric(labels, predictions) #평가지표

#데이터셋 Gaussian sampling
np.random.seed(0)

pts = list() # x,y
labels = list() # 10 0 ~9 
center_pts = np.random.uniform(-8.0, 8.0, (10,2))
for label, center_pts in enumerate(center_pts):
    for _ in range(100): 
        pts.append(center_pts + np.random.randn(*center_pts.shape)) #정규분포
        labels.append(label)

pts = np.stack(pts, axis=0).astype(np.float32) #list -> numpy array  gpu float 32
labels = np.stack(labels, axis = 0)

train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(1000).batch(32) #ram 


#모델 생성 
model = MyModel()

#손실함수, 최적화 알고리즘 
loss_object = tf.keras.losses.SparseCategoricalCrossentropy() #0,1,2,3, # one hot [ 0,0,1,0,] , [0,0,..]
optimizer = tf.keras.optimizers.Adam()
#Accuracy 
train_loss= tf.keras.metric.Mean(name = 'train_loss')
train_Accuracy = tf.keras.metrics.SaprseCategoricalAccuray(name = 'train_accuracy')

for epoch in range(epochs):
    for x, label in train_ds:
        train_step(model, x, label, loss_object, optimizer, train_loss, train_Accuracy)
    print(f'Ecpoch {epoch}, Loss:{train_loss.result()}, Acuracy{train_Accuracy.result()*100}')

#저장
np.savez_compressed('ch2_dataset.npz', inputs = pts, labels = labels)

W_h, b_h = model.d1.get_weights()
W_o, b_o = model.d2.get_weights()
W_h = np.transpose(W_h)
W_o = np.transpose(W_o)
np.savez_compressed('ch2_parameters.npz', W_h = W_h, b_h = b_h, W_o = W_o, W_h = W_h)



