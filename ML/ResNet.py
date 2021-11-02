import tensorflow as tf
import numpy as np

EPOCHS = 10

class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size): #channel(before), channel(after)
        super(ResidualUnit, self).__init__()
        #activation - preactivation 
        #weight
        #activation 
        #weight
        self.bn1 = tf.keras.layers.BatchNormalization()
        #activation
        self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        #activation
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding='same')

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding='same')
    def __call__(self, x, training = False, mask = None):
        h = self.bn1(x, training = training) #batch noramlization -> training 다르다
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(x, training = training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()
        #[16, 32, 32, 32 .., filter_out] # (16, 32) (32, 32) (32, filter_out)
        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))

    def __call__(self, x, training = False, mask = None):
        for unit in self.sequence:
            x = unit(x, training = training )
        return x

class Resnet(tf.keras.Model):
    def __init__(self):
        super(Resnet, self).__init__(self)
        self.conv1 = tf.keras.layers.Conv2D(8, (3, 3), padding = 'same', activation = 'relu') # 8*28*28

        self.res1 = ResnetLayer(8, (16, 16), (3, 3)) #16 * 28*28
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2)) # 16 * 14 * 14

        self.res2 = ResnetLayer(16, (32, 32), (3, 3)) # 32 * 14 * 14
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2))  #32 *  7* 7

        self.res3 = ResnetLayer(32, (64, 64), (3, 3)) # 64 *  7* 7 
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(10, activation ='softmax')
    def __call__(self, x, training = False, mask = None):
        x = self.conv1(x)

        x = self.res1(x, training=training)
        x = self.pool1(x)
        x = self.res2(x,training=training)
        x = self.pool2(x)
        x = self.res3(x,training=training)

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




#데이터셋

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test  = x_train/255.0, x_test/255.0

# print(x_train.shape) (60000, 28, 28) y_train.shape (60000, )
#(num_samples, 28, 28) -> (num_samples, 28, 28, 1)
x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_train[... , tf.newaxis].astype('float32')

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

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