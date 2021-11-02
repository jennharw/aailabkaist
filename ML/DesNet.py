import tensorflow as tf
import numpy as np
#데이터셋
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

EPOCHS = 10

class DenseUnit(tf.keras.Model):
    def __init__(self, filter_out, kernel_size):
        super(DenseUnit, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.concat = tf.keras.layers.Concatenate()
    def __call__(self, x, training = False, mask = None): # x: (Batch, H, W, Ch_in)
        h = self.bn(x)
        h = tf.nn.relu(h)
        h = self.conv(h)
        return self.concat([x, h]) #(Batch, H, W, ch_in + fil_out)


class DenseLayer(tf.keras.Model):
    def __init__(self, num_unit, growth_rate, kernel_size): #channel(before), c+k (growth rate) channel(after)
        super(DenseLayer, self).__init__()
        self.sequence = list()
        for i in range(num_unit):
            self.sequence.append(DenseUnit(growth_rate, kernel_size))
        # #activation
        # self.conv1_bn = tf.keras.layers.Conv2D(filter_in + k, (1, 1), padding='same') #bottle neck (1, 1)
        # self.conv1 = tf.keras.layers.Conv2D(filter_in + k, kernel_size, padding='same') #(3, 3)
        # #activation
        # self.conv2_bn = tf.keras.layers.Conv2D(filter_in + k*2, (1, 1), padding='same') #bottle neck (1, 1)
        # self.conv2 =tf.keras.layers.Conv2D(filter_in + k*2, kernel_size, padding='same')
        # self.conv3_bn = tf.keras.layers.Conv2D(filter_in + k*3, (1, 1), padding='same') #bottle neck (1, 1)
        # self.conv3 =tf.keras.layers.Conv2D(filter_in + k*3, kernel_size, padding='same')
        # self.conv4_bn = tf.keras.layers.Conv2D(filter_in + k*4, (1, 1), padding='same') #bottle neck (1, 1)
        # self.conv4 =tf.keras.layers.Conv2D(filter_in + k*4, kernel_size, padding='same')
    def __call__(self, x, training = False, mask = None):
        for unit in self.sequence:
            x = unit(x, training = training)
        return x

class TransitionLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(TransitionLayer, self).__init__()
        self.conv = tf.layers.Conv2D(filters, kernel_size, padding ='same')
        self.pool = tf.layers.MaxPool2D() #(2, 2)
    
    def __call__(self, x, training = False, mask = None):
        x = self.conv(x)
        return self.pool(x)

class DensNet(tf.keras.Model):
    def __init__(self):
        super(DensNet, self).__init__()
        self.conv1 = tf.keras.layers.conv2D(8 , (3, 3), padding = 'same', activation ='relu') #(28, 28, 1) -> (28, 28, 8)

        self.dl1 = DenseLayer(2, 4, (3, 3)) # (28, 28, 8+4+4)
        self.tr1 = TransitionLayer(16 , (3,3)) #(14, 14, 16) 

        self.dl2 = DenseLayer(2, 8, (3, 3)) # (14, 14, 16+8+8) 
        self.tr2 = TransitionLayer(32 , (3,3)) #(7,7,32) 
        self.dl3 = DenseLayer(2, 16, (3, 3)) # (7,7, 32+16+16) 

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(10, activation = 'softmax')
    def __call__(self, x, training = False, mask = None):
        x = self.conv1(x)

        x = self.dl1(x, training = training)
        x = self.tr1(x)

        x = self.dl2(x, training = training)
        x = self.tr2(x)

        x = self.dl3(x, training = training)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

#train
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

#test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training = False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)





#학습환경
model = Resnet()
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