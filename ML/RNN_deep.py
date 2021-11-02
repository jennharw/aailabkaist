import nltk
import itertools
#nltk.download("book")
import pandas as pd
import numpy as np
import sys
#tokenize
#remove infrequent words
#SOS, EOS
#metrics

vocabulary_size = 600
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SOS"
sentence_end_token = "EOS"

url = 'https://raw.githubusercontent.com/dennybritz/rnn-tutorial-rnnlm/master/data/reddit-comments-2015-08.csv'
df1 = pd.read_csv(url)
df1 = df1[:5000]
df1 = df1.apply(lambda x: nltk.sent_tokenize(x[0].lower()), axis = 1)
df2 = df1.apply(lambda x: "%s %s %s" % (sentence_start_token, x[0], sentence_end_token))
tokenized_sentences = [nltk.word_tokenize(sent) for sent in df2]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
vocab = word_freq.most_common(vocabulary_size -1 )
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
for i, sent in enumerate(tokenized_sentences):
  tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class vanillaRNN():
    def __init__(self, num_input, hidden_dim = 100, bptt_truncate = 4):
        self.num_input = num_input
        self.hidden_dim = hidden_dim
        self.bptt_truncate = 4

        self.U = np.random.uniform(-np.sqrt(1./num_input), np.sqrt(1./num_input), (hidden_dim, num_input)) #Wxh
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim)) #Whh  
        
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (num_input, hidden_dim))  #Vhy num_output

    def forward_propagation(self, x): #__call__(self, x):
        #Forward propagation
        T = len(x)
        #S, O 저장하려고 matrix
        #s0 를 0으로 
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # hn = np.tanh(Wxh * input + Whh * hn-1)
        # y = np.softmax(Vhy * hn)
        o = np.zeros((T, self.num_input)) #num_output
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]
    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis = 1)
    
    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_word_prediction = o[np.arange(len(y[i])), y[i]] # o (600, len(y) 73)
            L += -1 * np.sum(np.log(correct_word_prediction))
        return L
    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y)/N
    
    def bptt(self, x, y): # SGD U, V, W 찾기 
        T = len(y)
        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1. #delta_o y_hat - y
        #[array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    #    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    #    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    #    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    #    68, 69, 70, 71]), [480, 599, 8, 26, 599, 60, 599, 187, 21, 3, 184, 205, 165, 4, 599, 584, 599, 76, 87, 599, 444, 3, 599, 8, 119, 8, 5, 115, 72, 270, 45, 60, 29, 6, 410, 145, 177, 66, 34, 19, 26, 97, 599, 599, 599, 14, 168, 247, 106, 100, 599, 18, 107, 5, 34, 19, 82, 599, 17, 8, 599, 18, 86, 66, 185, 234, 159, 35, 599, 17, 2, 1]]
        for t in range(T):
            dLdV += np.outer(delta_o[t], s[t].T) #V는 t에서만  
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]: #W, U는 더해줘야
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])              
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
            
    #gradient (f(x + h) - f(x)) /h,  h = 0.001, error_threshold = 0.01 => (L(x+h) - L(x-h))/2h
    #곱으로 나타낼 수 - chain rule
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

model = vanillaRNN(vocabulary_size)
o, s =  model.forward_propagation(X_train[10])
model.sgd_step(X_train[10], y_train[10], 0.005)


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch): #100번 update * len(y_train)
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
           # time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(": Loss after num_examples_seen=%d epoch=%d: %f" % (num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                #print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


model = vanillaRNN(vocabulary_size) #hidden_dim 100
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)

new_sentence = [word_to_index[sentence_start_token]] 
print("new_sentence")
print(new_sentence)
print(len(new_sentence))
next_word_probs = model.forward_propagation(new_sentence)
print("next_word_probs")
print(len(next_word_probs))
print(len(next_word_probs[0]))
sampled_word = word_to_index[unknown_token]
print("Smaple ")
print(sampled_word)
print(np.random.multinomial(n=1, pvals = next_word_probs[0].tolist()[0]))
samples = np.random.multinomial(n=1, pvals = next_word_probs[0].tolist()[0]) 
sampled_word = np.argmax(samples)
print(sampled_word)
# samples = np.random.multinomial(1, next_word_probs[0]) #o, s
# sampled_word = np.argmax(samples)
# print("chagne")
# print(sampled_word)


#문장 생성하기
def generate_sentence(model):
    new_sentence = [word_to_index[sentence_start_token]] #"SOS"
    #repeat until we get end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]: #EOS
        next_word_probs = model.forward_propagation(new_sentence) #[0] # [o, s]
        sampled_word = word_to_index[unknown_token] #599
        #we dont want unknownwords #255
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(n=1, pvals = next_word_probs[0].tolist()[0]) 
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str


num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print(" ".join(sent))

        

#GRU
# class GRU():
#     def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):

#         self.word_dim = word_dim
#         self.hidden_dim = hidden_dim
#         self.bptt_truncate = bptt_truncate

#         # Initialize the network parameters
#         E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
#         U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
#         W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
#         V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
#         b = np.zeros((6, hidden_dim))
#         c = np.zeros(word_dim)

#     def __call__(self, x_t, s_t1_prev, s_t2_prev):
#         E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

#         x_e = E[:,x_t]
            
#         # GRU Layer 1
#         z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
#         r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
#         c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
#         s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
#         #Final output calculation
#             # Theano's softmax returns a matrix with one row, we only need the row
#         o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

#         return [o_t, s_t1, s_t2]    
       
#     def gradeint(self):
#         dE = T.grad(cost, E)
#         dU = T.grad(cost, U)
#         dW = T.grad(cost, W)
#         db = T.grad(cost, b)
#         dV = T.grad(cost, V)
#         dc = T.grad(cost, c)

#     def optimizer(self):
#         #Rmsprop의 기본적인 아이디어는, 이전 gradient들의 합에 따라 파라미터별로 learning rate을 조정하는 것입니다. 직관적으로 이해해보면, 자주 등장하는 특징(feature)들은 작은 learning rate를 갖게 되고 (gradient들의 합이 작기 때문에), 드문드문 등장하는 특징들은 큰 learning rate를 갖게 됩니다.
#         # #구현은 상당히 간단합니다. 각 파라미터마다 캐시 변수를 두고, gradient descent가 진행될 때 아래와 같이 파라미터와 캐시를 업데이트하면 됩니다 (W에 대한 예시):
#         cacheW = decay * cacheW + (1 - decay) * dW ** 2

#         W = W - learning_rate * dW / np.sqrt(cacheW + 1e-6)
    # def prediction(self):
    #     prediction = T.argmax(o, axis=1)