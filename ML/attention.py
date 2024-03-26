import random 
import tensorflow as tf
from konlpy.tag import Okt
import numpy as np

EPOCHS = 200
NUM_WORDS = 2000

#Attention Network 
##Wq, Wk, Wv

#데이터셋 (batch, word_nums, len)
dataset_file = 'chatbot_data.csv'
okt = Okt()
with open('chatbot_data.csv', 'r') as file:
  lines = file.readlines()
  seq = [' '.join(okt.morphs(line)) for line in lines]

questions = seq[::2]
answers = ['\t ' + line for line in seq[1::2]]

num_sample = len(questions)

perm = list(range(num_sample))
random.seed(0)
random.shuffle(perm)

train_q = list()
train_a = list()
test_q = list()
test_a = list()

for idx, qna in enumerate(zip(questions, answers)):
  q, a = qna
  if perm[idx] > num_sample//5:
      train_q.append(q)
      train_a.append(a)
  else:
      test_q.append(q)
      test_a.append(a)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = 2000,
                                                  filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
tokenizer.fit_on_texts(train_q + train_a)
train_q_seq = tokenizer.texts_to_sequences(train_q)
train_a_seq = tokenizer.texts_to_sequences(train_a)

test_q_seq = tokenizer.texts_to_sequences(test_q)
test_a_seq = tokenizer.texts_to_sequences(test_a)

x_train = tf.keras.preprocessing.sequence.pad_sequences(train_q_seq, value = 0, padding = 'pre', maxlen=64)
y_train =tf.keras.preprocessing.sequence.pad_sequences(train_a_seq, value = 0, padding = 'post', maxlen=65)
y_train_shifted = np.concatenate([np.zeros((y_train.shape[0], 1)), y_train[: , 1:]], axis = 1)

x_test = tf.keras.preprocessing.sequence.pad_sequences(test_q_seq,
                                                       value=0,
                                                       padding='pre',
                                                       maxlen=64)
y_test = tf.keras.preprocessing.sequence.pad_sequences(test_a_seq,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=65)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32).prefetch(1024)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1).prefetch(1024) #메모리 적재 

#encoder
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64) # x (num_words, 64 len_sentence)
        self.lstm = tf.keras.layers.LSTM(512, return_sequences = True,  return_state = True) #h output_dim 512 (512, 64)
    def __call__(self, x, training = False, mask = None):
        x = self.emb(x)
        H, h, c = self.lstm(x) #hidden state, cell state - Query s0 , # 출력이 모두 나오도록 모든 hidden state를 sequences형태로 
        return H, h, c #attention key, value / lstm layer 연결
#decoder
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 64) # y (num_words, 64) 
        self.lstm = tf.keras.layers.LSTM(512, return_sequences = True, return_state = True) # (512, 64)
        self.att = tf.keras.layers.Attention() #query : s0  \ key, value : H of the encoder (512, 64)
        self.dense = tf.keras.layers.Dense(NUM_WORDS, activation='softmax') #ouptut num_words (num_words, 64)

    def __call__(self, inputs, training = False, mask = None):
        y_, s0, c0, H = inputs # shifted_output,LSTM(y_, s0, c0)  hidden (key value)  
        y_ = self.emb(y_) #LSTM 부분
        S, h, c =  self.lstm(y_, initial_state=[s0, c0]) #encoder 의 마지막 , sos
        #S(모든 hidden state)를 쿼리로 사용한다
        #그러나 한 타임 앞선 시간, s0가 맨 앞으로
        #s0은 2차원 length 가 1인, 3차원으로 확장 tf.enwaxis 1짜리 추가
        #S마지막 hidden state 는 배제 1 + (64-1) 
        S_ = tf.concat([s0[:,tf.newaxis, :], S[:, :-1, :]], axis = 1) #why? new axis  #new axis 옆으로  , S 마지막 제외
        A =  self.att([S_, H]) # 이전꺼? S_ # key value 같은 것 H, f
        y = tf.concat([S, A], axis =-1) #(512 + 512, 64)
        return self.dense(y), h, c #(512, 64)
        

#model seqtoseq
class Seq2seq(tf.keras.Model):
    def __init__(self, sos, eos):
        super(Seq2seq, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.sos = sos
        self.eos = eos

    def __call__(self, inputs, training = False, mask = None):
        if training is True:
            x, y = inputs
            H, h, c = self.enc(x)
            y, _, _ = self.dec((y, h, c, H)) #Decoder inputs - y, s0, c0, H 
            return y
        #softmax(S * h(1 ~ ))
        #a sigma
        #argmax v

        else:
            x = inputs
            H, h, c = self.enc(x)
            y = tf.convert_to_tensor(self.sos)
            y = tf.reshape(y, (1, 1)) # batch 1 , (1, 1)

            seq = tf.TensorArray(tf.int32, 64)

            for ids in tf.range(64):
                y, h, c = self.dec([y, h, c, H])
                y = tf.cast(tf.argmax(y, axis =-1), dtype = tf.int32)
                y = tf.reshape(y, (1, 1)) #test일 때 넣어줘야
                seq = seq.write(ids, y)

                if y == self.eos:
                    break
            return tf.reshape(seq.stack(), (1, 64)) # 문장길이 (1, 1, 64)


#학습, 테스트 루프 정의
#training loop
@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    output_labels = labels[:, 1:]
    shifted_labels = labels[:, :-1] #EOS 제거
    with tf.GradientTape() as tape:
        predictions = model([inputs, shifted_labels], training=True)
        loss = loss_object(output_labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(output_labels, predictions)

@tf.function
def test_step(model, inputs):
    return model(inputs, training = False)




#학습환경
model = Seq2seq(sos = tokenizer.word_index['\t'] , eos = tokenizer.word_index['\n'])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.losses.SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

#
for epoch in range(EPOCHS):
    for seqs, labels in train_ds :
        train_step(model, seqs, labels, loss_object, optimizer, train_loss, train_accuracy)

    print(f'epoch : {epoch + 1} , train_loss : {train_loss.result()}, train_accuracy : {train_accuracy.result()}')


#test
for test_seq, test_labels in test_ds:
    prediction = test_step(model, test_seq)
    test_text = tokenizer.sequences_to_texts(test_seq.numpy())
    gs_text = tokenizer.sequences_to_texts(test_labels.numpy())
    pre_text = tokenizer.sequences_to_texts(prediction.numpy())
    print(f"- \n q: {test_text} \n a:{gs_text} \n p: {pre_text}")