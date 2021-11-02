import scipy.io as sio
data = sio.loadmat('sentimentdataset.mat')
bagofword = data['bagofword']
sentiment = data['sentiment']
sentiment = sentiment.astype(int)
words = data['word']

