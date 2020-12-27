import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import tensorflow as tf
import torch.nn.functional as F

with open("../input/data.json") as fp:
    data = json.load(fp)

#把review和rating放到兩個list裏
reviews = []
ratings = []

for data_point in data:
    review = data_point["review"]
    review = review.lower()
    #remove punctuation
    review = re.sub(r'[^\w\s]', ' ', review)
    reviews.append(review)
    rating = int(data_point["rating"])-1
    ratings.append(rating)


print ('Number of reviews :', len(reviews))
print(reviews[15])
print(ratings[0:15])
#print(ratings.size())
ratings = np.array(ratings, dtype=int)

all_words = ' '.join(reviews)
# create a list of words
word_list = all_words.split()
# Count all the words using Counter Method
count_words = Counter(word_list)
len_words = len(word_list)

len_counted_words = len(count_words)

print("len_words = ", len_words)
print("len_counted_words = ", len_counted_words)
sorted_words = count_words.most_common(len_words)
# Code words into numbers
word_to_num = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
reviews_num = []
for review in reviews:
    num = [word_to_num[w] for w in review.split()]
    reviews_num.append(num)
print (reviews_num[0:3])

reviews_len = [len(x) for x in reviews_num]
pd.Series(reviews_len).hist()
plt.show()

reviews_pad = np.zeros((len(reviews_num), 1000), dtype = int)

for i, review in enumerate(reviews_num):
    review_len = len(review)

    if review_len <= 1000:
        zeroes = list(np.zeros(1000-review_len))
        new = zeroes+review
    elif review_len > 1000:
        new = review[0:1000]

    reviews_pad[i,:] = np.array(new)

#Split training set, valid set and testing set

train_size = 0.8
train_x = reviews_pad[0:int(train_size*len(reviews_pad))]
train_y = ratings[0:int(train_size*len(ratings))]
val_test_x = reviews_pad[int(train_size*len(reviews_pad)):]
val_test_y = ratings[int(train_size*len(reviews_pad)):]
val_x = val_test_x[0:int(len(val_test_x)*0.5)]
val_y = val_test_y[0:int(len(val_test_y)*0.5)]
test_x = val_test_x[int(len(val_test_x)*0.5):]
test_y = val_test_y[int(len(val_test_y)*0.5):]

print(len(train_x))
print(len(train_y))
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
val_x = torch.tensor(val_x)
val_y = torch.tensor(val_y)
test_x = torch.tensor(test_x)
test_y = torch.tensor(test_y)
print(train_x.size())
print(val_x.size())
print(train_x.size())
print(val_y.size())

#train_data = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_data = Data.TensorDataset(train_x, train_y)
valid_data = Data.TensorDataset(val_x, val_y)
test_data = Data.TensorDataset(test_x, test_y)


batch_size = 64

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        #out = F.log_softmax(out)
        
        # reshape to be batch_size first
        out = out.view(batch_size, -1)
        out = out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        
        return out, hidden
    
    
    #def init_hidden(self, batch_size):
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            #hidden = (torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).cuda(),
                      #torch.zeros(self.n_layers, self.batch_size, self.hidden_dim).cuda())
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                         weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
            #W1 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
            #W2 = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
            #hidden = (W1, W2)
        #else:
            #hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      #weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

# Instantiate the model w/ hyperparams
vocab_size = len(word_to_num)+1 # +1 for the 0 padding
output_size = 10
embedding_dim = 400
hidden_dim = 512
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# loss and optimization functions
lr=0.001

#criterion = nn.BCELoss()
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params

num_epochs = 5 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net = net.to(device)

#net.train()
# train for some number of epochs
print("start training")
for epoch in range(num_epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)
    # batch loop
    for i, (inputs, labels) in enumerate(train_loader):

        if(train_on_gpu):
            inputs, labels = inputs.to(device), labels.to(device)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        #h = net.init_hidden()
        batch_size = inputs.size(0)
        h = net.init_hidden(batch_size = batch_size)
        
        #h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.cuda.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        #labels = labels.unsqueeze(0)
        loss = criterion(output, labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step {}, Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, loss.item()))
print("start predicting")
batch_size = 64                  
val_h = net.init_hidden(batch_size)
right_num = list()
total_num = list()
for i, (vali_x, vali_y) in enumerate(valid_loader):
    vali_x = vali_x.to(device)
    batch_size = vali_x.size(0)
    val_h = net.init_hidden(batch_size = batch_size)
    net.zero_grad()
    pred_y, val_h = net(vali_x, val_h)
    pred_y = pred_y.cpu().data.numpy()
    
    pred_y = np.rint(pred_y)
    
            # print(vali_output.size())
    
    #_, pred_y = torch.max(pred_y.data, 0)
            # print(pred_y.shape)
            # print(float((pred_y == vali_y.numpy()).sum()), len(pred_y))
            # accuracy = float((pred_y == vali_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    right_num.append(float((pred_y == vali_y.numpy()).sum()))
    #right_num.append(int(torch.sum(tf.equal(pred_y == vali_y))))
    total_num.append(len(pred_y))
    if (i+1) % 10 == 0:
        print ('val step', i+1)
accuracy = 100*sum(right_num) / sum(total_num)
#if accuracy > best_accuracy:
    #best_accuracy = accuracy

print('validation accuracy: %.2f' % accuracy, 'Total right: ', sum(right_num), '| Total: ', sum(total_num))
print("start predicting")
batch_size = 64                   
test_h = net.init_hidden(batch_size)
right_num = list()
total_num = list()
for i, (test_x, test_y) in enumerate(test_loader):
    test_x = test_x.to(device)
    batch_size = test_x.size(0)
    test_h = net.init_hidden(batch_size = batch_size)
    net.zero_grad()
    pred_y, test_h = net(test_x, test_h)
    pred_y = pred_y.cpu().data.numpy()
    
    pred_y = np.rint(pred_y)
    
            # print(vali_output.size())
    
    #_, pred_y = torch.max(pred_y.data, 0)
            # print(pred_y.shape)
            # print(float((pred_y == vali_y.numpy()).sum()), len(pred_y))
            # accuracy = float((pred_y == vali_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    right_num.append(float((pred_y == test_y.numpy()).sum()))
    #right_num.append(int(torch.sum(tf.equal(pred_y == vali_y))))
    total_num.append(len(pred_y))
    if (i+1) % 10 == 0:
        print ('test step', i+1)
accuracy = 100*sum(right_num) / sum(total_num)
#if accuracy > best_accuracy:
    #best_accuracy = accuracy

print('test accuracy: %.2f' % accuracy, 'Total right: ', sum(right_num), '| Total: ', sum(total_num))
