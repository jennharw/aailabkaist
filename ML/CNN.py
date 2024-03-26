#CNN

import tensorflow as tf
import numpy as np

EPOCHS = 10
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
print(y_train)
train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
#모델

class ConvNet(tf.keras.Model):
    def __init__(self):
        conv2d = tf.keras.layers.Conv2D
        maxpool = tf.keras.layers.MaxPool2D
        super(ConvNet, self).__init__()
        self.sequence = list()
        self.sequence.append(conv2d(16, (3, 3), padding = 'same', activation = 'relu'))  # 28,28,16 
        self.sequence.append(conv2d(16, (3, 3), padding = 'same', activation = 'relu')) # 28,28,16 
        self.sequence.append(maxpool((2,2))) #14,14,16
        self.sequence.append(conv2d(32, (3, 3), padding = 'same')) # 14,14,32
        self.sequence.append(conv2d(32, (3, 3), padding = 'same')) # 14,14,32
        self.sequence.append(maxpool(2,2)) # 7,7,32
        self.sequence.append(conv2d(64, (3, 3), padding = 'same')) #7,7,64
        self.sequence.append(conv2d(64, (3, 3), padding = 'same')) #7,7,64
        self.sequence.append(tf.keras.layers.Flatten()) #1568
        self.sequence.append(tf.keras.layers.Dense(2048 ,activation = 'relu'))
        self.sequence.append(tf.keras.layers.Dense(10 ,activation = 'softmax'))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

#train
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

#test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

#데이터셋

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test  = x_train/255.0, x_test/255.0

# print(x_train.shape) (60000, 28, 28) y_train.shape (60000, )
#(num_samples, 28, 28) -> (num_samples, 28, 28, 1)
x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_train[... , tf.newaxis].astype('float32')

# train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#학습환경
model = ConvNet()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name ="train_accuracy")

test_loss= tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = "test_accuracy")

#학습
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
    for test_images, test_labels in test_ds:
        test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
    print(f'Epoch {epoch + 1}, Loss:{train_loss.result()}, Accuracy:{train_accuracy.result()*100}, test Loss: {test_loss.result()}, test accuracy: {test_accuracy.result()*100}')