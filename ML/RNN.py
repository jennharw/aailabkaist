import tensorflow as tf

EPOCHS = 10
NUM_WORDS = 10000

imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = NUM_WORDS)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, value = 0, padding = 'pre', maxlen = 32)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, value = 0, padding = 'pre', maxlen = 32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(32)


#MODEL
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)
        self.rnn = tf.keras.layers.SimpleRNN(32) #LSTM(32), GRU(32)
        self.dense = tf.keras.layers.Dense(2, activation = 'softmax')
    def __call__(self, x, training = False, mask = None):
        x = self.emb(x)
        x = self.rnn(x)
        return self.dense(x)

#train
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True)
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
model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name ="train_accuracy")

test_loss= tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = "test_accuracy")

#학습
for epoch in range(EPOCHS):
    for seqs, labels in train_ds:
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)
    for test_seqs, test_labels in test_ds:
        test_step(model, test_seqs, test_labels, loss_object, test_loss, test_accuracy)
    print(f'Epoch {epoch + 1}, Loss:{train_loss.result()}, Accuracy:{train_accuracy.result()*100}, test Loss: {test_loss.result()}, test accuracy: {test_accuracy.result()*100}')