#transformer translation

import posix
import random
import numpy as np
import tensorflow as tf
#import konlpy.tag import Okt

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Lambda, Layer, Embedding, LayerNormalization

import os

EPOCHS = 200
NUM_WORDS = 2000

"""
MASK 표현
comp = np.ones((5,5))
mask = tf.fill((5, 5), -np.inf)
mask = tf.linalg.band_part(mask, 0, -1)
mask = tf.linalg.set_diag(mask, tf.zeros((5)))
comp += mask
"""

#Dot-Scaled Attention
##C = softmax(K_T * Q / dk), a = CT * V
class DotScaledAttention(Layer):
    def __init__(self, d_emb, d_reduced, masked  = False):
        super().__init__()
        self.q = Dense(d_reduced, input_shape = (-1, d_emb))  # (batch, 128, 16) = Wq, Wk, Wv
        self.k = Dense(d_reduced, input_shape = (-1, d_emb))
        self.v = Dense(d_reduced, input_shape = (-1, d_emb))

        self.scale = Lambda(lambda x : x/np.sqrt(d_reduced)) # dk
        self.masked = masked
    def __call__(self, x, training = False, mask = None): # x inputs  : (q, k, v)
        q = self.scale(self.q(x[0]))
        k = self.k(x[1])
        v = self.v(x[2])
        #inner product
        k_T = tf.transpose(k, perm = [0,2,1]) # batch는 유지하고, 2, 1 로 transpose
        comp = tf.matmul(q, k_T)

        if self.masked:
            length = tf.shape(comp)[-1] #128
            mask = tf.fill((length, length), -np.inf) #softmax 하면 없어지도록
            mask = tf.linalg.band_part(mask, 0, -1) 
            mask = tf.linalg.set_diag(mask, tf.zeros((length)))
            comp += mask

        comp = tf.nn.softmax(comp, axis = -1)
        return tf.matmul(comp, v) # -> attention value #(batch, len, emb)

#Multi Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, num_head, d_emb, d_reduced, masked=False):
        super().__init__()
        self.attention_list = list() # 여러개의 attention
        for _ in range(num_head):
            self.attention_list.append(DotScaledAttention(d_emb, d_reduced, masked)) # (16, 16, 4) | (batch, 128, 16) -> (batch, 128, 16)
        self.linear = Dense(d_emb, input_shape = (-1, num_head * d_reduced)) #(16)
    def __call__(self, x, training = False, mask = None):
        attention_list = [a(x, training)  for a in self.attention_list] 
        concat = tf.concat(attention_list, axis = -1) #(batch, 128, 16 * 4)
        return self.linear(concat) #(batch, 128, 16)


#Encoder
class Encoder(Layer):
    def __init__(self, num_head, d_reduced): # input shape - dim
        super().__init__()
        #embedding
        #positional Encoding
        #Encoder
        #Multi head Attention, Add & Norm , Feed Forward (Dense) , Add & Norm = > outputs
        self.num_head = num_head
        self.d_r = d_reduced
    def build(self, input_shape):
        self.multi_attention = MultiHeadAttention(self.num_head, input_shape[-1], self.d_r) # (4, 16, 16) #(batch, 128, 16) -> (batch, 128, 16)
        self.layer_norm1 = LayerNormalization(input_shape=input_shape)
        self.dense1 = Dense(input_shape[-1] * 4, input_shape = input_shape, activation = 'relu') 
        self.dense2 = Dense(input_shape[-1], input_shape = self.dense1.compute_output_shape(input_shape))
        self.layer_norm2 = LayerNormalization(input_shape=input_shape)
        super().build(input_shape)
    def call(self, x, training = None, mask = None): #build __ error
        h = self.multi_attention((x, x, x)) # q, k, v , encoder (self Attention 구조 )
        ln1 = self.layer_norm1(x + h) # skip connection

        h = self.dense2(self.dense1(ln1))
        return self.layer_norm2(h + ln1)

    def compute_output_shape(self, input_shape):
        return input_shape

#Decoder
## y_(shifted outputs) -> embedding(positional embedding) -> self-attention(Multihead attention) + add*norm
## Multihead attention (query - 위의 attention 값, Key Value - Encoder 의 출력값 )
##Dense, Add norm 
##Linear, softmax
class Decoder(Layer):
    def __init__(self, num_head, d_reduced):
        super().__init__()
        self.num_head = num_head
        self.d_r = d_reduced
    def build(self, input_shape):
        self.self_attention = MultiHeadAttention(self.num_head, input_shape[0][-1], self.d_r,   masked=True) #num_head concat -> lenear
        self.layer_norm1 = LayerNormalization(input_shape=input_shape)
        self.multi_attention = MultiHeadAttention(self.num_head, input_shape[0][-1], self.d_r)              
        self.layer_norm2 = LayerNormalization(input_shape=input_shape)
        self.dense1 = Dense(input_shape[0][-1] * 4, input_shape = input_shape[0], activation = 'relu')
        self.dense2 = Dense(input_shape[0][-1] , input_shape = self.dense1.compute_output_shape(input_shape[0]))
        self.layer_norm3 = LayerNormalization(input_shape=input_shape)
        super().build(input_shape)
    def call(self, inputs, training = None, mask = None): #inputs (y, context - Encoder의 출력값)  #build __ error, 먼저 불려서 
        x, context = inputs
        h = self.self_attention((x, x, x)) # (batch, 128, 16)
        ln1 = self.layer_norm1(x + h) # skip connection

        h = self.multi_attention((ln1, context, context)) # Query, (key value)
        ln2 = self.layer_norm2(ln1 + h)

        h = self.dense2(self.dense1(ln2))
        return self.layer_norm3(h + ln2) # (batch, 128, 16)

    def compute_ouptut_sahpe(self, input_shape):
        return input_shape
    
class PositionalEncoding(Layer):
    def __init__(self, max_len, d_emb):
        super().__init__()
        self.sinusoidal_encoding = np.array([self.get_positional_angle(pos, d_emb) for pos in range(max_len)])
        self.sinusoidal_encoding[:, 0::2] = np.sin(self.sinusoidal_encoding[:, 0::2])
        self.sinusoidal_encoding[:, 1::2] = np.cos(self.sinusoidal_encoding[:, 1::2])
        self.sinusoidal_encoding = tf.cast(self.sinusoidal_encoding, dtype = tf.float32)


    def call(self, x, training = None, mask = None):
        return x + self.sinusoidal_encoding[:tf.shape(x)[1]]
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_angle(self, pos, dim, d_emb):
        return pos / np.power(10000, 2 * (dim//2) / d_emb)
    def get_positional_angle(self, pos, d_emb):
        return [self.get_angle(pos, dim, d_emb) for dim in range(d_emb)]
#positional Encoding - cos, sin

#Transformer Architecture  NUM_WORDS, NUM_WORDS, 128, 16, 16, 2, 2, 4 | transformer.fit([x_train, y_train_shifted], y_train, batch_size = 5, epochs = EPOCHS)
class Transformer(Model):
    def __init__(self, src_vocab, dst_vocab, max_len, d_emb, 
                    d_reduced, n_enc_layer, n_dec_layer, num_head):
        super().__init__()
        self.enc_emb = Embedding(src_vocab, d_emb) 
        self.dec_emb = Embedding(dst_vocab, d_emb)  
        self.pos_enc = PositionalEncoding(max_len, d_emb)
        self.encoder = [Encoder(num_head, d_reduced) for _ in  range(n_enc_layer)] # multi layer 병렬 구조 (같은 input shape, output shape 나와야)
        self.decoder = [Decoder(num_head, d_reduced) for _ in range(n_dec_layer)]

        self.dense = Dense(dst_vocab, input_shape = (-1, d_emb))
    def call(self, inputs, training = False, mask = None): #inputs (src_sentence, dst_sentence_shifted)
        #Encoder 입력, Decoder 입력
        src_sent, dst_sent_shifted = inputs

        h_enc = self.pos_enc(self.enc_emb(src_sent)) # x (batch, 128 length, 2000) -> (batch, 128, 16)
        #Encoder
        for enc in self.encoder:
            h_enc = enc(h_enc) 

        h_dec = self.pos_enc(self.dec_emb(dst_sent_shifted)) # y (batch, 128, 2000) -> (batch, 128, 16)
        for dec in self.decoder:
            h_dec = dec([h_dec, h_enc]) # y, context
        return tf.nn.softmax(self.dense(h_dec), axis = -1) # (batch, 128, 2000) -> softmax



#Dataset
okt = Okt()

with open('chatbot_data.csv', 'r') as file:
  lines = file.readlines()
  seq = [' '.join(okt.morphs(line)) for line in lines]

import numpy as np
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
#fit training
transformer = Transformer(NUM_WORDS, NUM_WORDS, 128, 16, 16, 2, 2, 4) 

transformer.compile(optimizer = 'adam', 
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
transformer.fit([x_train, y_train_shifted], y_train, batch_size = 5, epochs = EPOCHS)