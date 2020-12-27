from __future__ import print_function

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


import logging
import logging.config

import psutil

#print(psutil.cpu_percent())
#print(psutil.virtual_memory())  # physical memory usage
#print('memory % used:', psutil.virtual_memory()[2])

import gc

#b_partial = True
b_partial = False

i_partial_count = 5000

#b_do_kaggle = False
b_do_kaggle = True

#b_do_colab = True
b_do_colab = False



b_use_new_data_set = True

#b_use_new_data_IMDB_or_YELP = True
b_use_new_data_IMDB_or_YELP = False


i_batch_size = 64

i_output_size = 0

if b_use_new_data_IMDB_or_YELP:
    i_output_size = 10
else:
    i_output_size = 5


s_root = ""


s_log_config_fn = "logging.conf"


if b_do_colab:

    s_root = "/content/drive/My Drive/gcolab/comp511prj4/"
    s_log_config_fn = s_root + s_log_config_fn



if b_do_kaggle:
    s_log_config_fn = "../input/group36-proj4-config/" + s_log_config_fn
    
    print(os.listdir("../input/"))
    
    
    

#logging.config.fileConfig(s_log_config_fn)


logger = logging.getLogger('Project4Group36')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")


print(psutil.virtual_memory())  # physical memory usage
print('memory % used before load dataset:', psutil.virtual_memory()[2])


str_input_folder = "../../../"

if b_do_kaggle:
    str_input_folder = "../input/yelp-imdb-multi-class-v2/datasets/"

if b_do_colab:
    str_input_folder = "/content/drive/My Drive/gcolab/dataset/datasets/"



if b_use_new_data_set:
    
    if b_use_new_data_IMDB_or_YELP:
        str_json_fn_training = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
    else:   
        str_json_fn_training = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"

print(str_json_fn_training)


data = None


with open(str_json_fn_training) as fp:
    data = json.load(fp)


if b_partial:
    data = data[0:i_partial_count]



print(psutil.virtual_memory())  # physical memory usage
print('memory % used after load dataset:', psutil.virtual_memory()[2])


#把review和rating放到兩個list裏
reviews = []
ratings_fs = []

for data_point in data:
    review = data_point["review"]


    review = review.lower()
    #remove punctuation
    review = re.sub(r'[^\w\s]', ' ', review)
    reviews.append(review)
    rating = int(data_point["rating"])-1
    ratings_fs.append(rating)


print ('Number of reviews :', len(reviews))
print(reviews[15])
print(ratings_fs[0:15])
#print(ratings.size())
ratings = np.array(ratings_fs, dtype=int)

all_words = ' '.join(reviews)




del ratings_fs


gc.collect()



print(psutil.virtual_memory())  # physical memory usage
print('memory % used after 1st stage cleanup:', psutil.virtual_memory()[2])



# create a list of words
word_list = all_words.split()
# Count all the words using Counter Method
count_words = Counter(word_list)
len_words = len(word_list)

len_counted_words = len(count_words)

print("len_words = ", len_words)
print("len_counted_words = ", len_counted_words)

print(psutil.virtual_memory())  # physical memory usage
print('memory % used after Counter:', psutil.virtual_memory()[2])



del all_words
del word_list


gc.collect()


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after 2nd stage cleanup:', psutil.virtual_memory()[2])




i_max_pick_size = 200000

i_max_pick_size = len_counted_words

sorted_words = count_words.most_common(i_max_pick_size)

sorted_word_list = [w for i, (w,c) in enumerate(sorted_words)]

set_sorted_word = set(sorted_word_list)

# Code words into numbers
word_to_num = {w:i+1 for i, (w,c) in enumerate(sorted_words)}




reviews_num = []
for review in reviews:
    #num = [word_to_num[w] for w in review.split()]
    num = []
    
    for w in review.split():
        if w in set_sorted_word:
            num.append(word_to_num[w])
    
    reviews_num.append(num)
print (reviews_num[0:3])

reviews_len = [len(x) for x in reviews_num]
#pd.Series(reviews_len).hist()
#plt.show()



logger.info("after data pre-process. ")


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after data pre-process:', psutil.virtual_memory()[2])

i_word_to_num_len = len(word_to_num)


del count_words
del sorted_words
del sorted_word_list
del set_sorted_word
del reviews

del word_to_num


gc.collect()


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after 3rd stage cleanup:', psutil.virtual_memory()[2])

i_len_according_to_hist = 0

if b_use_new_data_IMDB_or_YELP:
    i_len_according_to_hist = 1000
else:
    i_len_according_to_hist = 600

reviews_pad = np.zeros((len(reviews_num), i_len_according_to_hist), dtype = int)

for i, review in enumerate(reviews_num):
    review_len = len(review)

    if review_len <= i_len_according_to_hist:
        zeroes = list(np.zeros(i_len_according_to_hist - review_len))
        new = review + zeroes
    elif review_len > i_len_according_to_hist:
        new = review[0:i_len_according_to_hist]

    reviews_pad[i,:] = np.array(new)


#rp_size_mb = reviews_pad.memory_usage().sum() / 1024 / 1024
#print("Test memory size: %.2f MB" % rp_size_mb)
# Test memory size: 1879.24 MB

#reviews_pad = reviews_pad.replace(0, np.nan).to_sparse()
#rp_sparse_size_mb = reviews_pad_sparse.memory_usage().sum() / 1024 / 1024
#print("Test sparse memory size: %.2f MB" % rp_sparse_size_mb)


#Split training set, valid set and testing set

train_size = 0.8

train_x = reviews_pad[0:int(train_size*len(reviews_pad))]
train_y = ratings[0:int(train_size*len(ratings))]

val_test_x = reviews_pad[int(train_size*len(reviews_pad)):]
val_test_y = ratings[int(train_size*len(reviews_pad)):]


#val_x = reviews_pad[int(train_size*len(reviews_pad)):]
#val_y = ratings[int(train_size*len(reviews_pad)):]

val_x = val_test_x[0:int(len(val_test_x)*0.5)]
val_y = val_test_y[0:int(len(val_test_y)*0.5)]

test_x = val_test_x[int(len(val_test_x)*0.5):]
test_y = val_test_y[int(len(val_test_y)*0.5):]


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after data prepare:', psutil.virtual_memory()[2])


del reviews_num
del reviews_pad
del ratings

del val_test_x
del val_test_y


gc.collect()


print(psutil.virtual_memory())  # physical memory usage
print('memory % used before create tensors:', psutil.virtual_memory()[2])


print(len(train_x))
print(len(train_y))




train_x_t = torch.tensor(train_x)
train_y_t = torch.tensor(train_y)

val_x_t = torch.tensor(val_x)
val_y_t = torch.tensor(val_y)

test_x_t = torch.tensor(test_x)
test_y_t = torch.tensor(test_y)




print(train_x_t.size())
print(val_x_t.size())
print(train_y_t.size())
print(val_y_t.size())

print(psutil.virtual_memory())  # physical memory usage
print('memory % used after create tensors:', psutil.virtual_memory()[2])



del train_x
del train_y
del val_x
del val_y
del test_x
del test_y



gc.collect()


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after 5th stage cleanup:', psutil.virtual_memory()[2])



#train_data = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
train_data = Data.TensorDataset(train_x_t, train_y_t)
valid_data = Data.TensorDataset(val_x_t, val_y_t)
test_data = Data.TensorDataset(test_x_t, test_y_t)


batch_size = i_batch_size

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
vocab_size = i_word_to_num_len + 1 # +1 for the 0 padding
output_size = i_output_size
embedding_dim = 400
hidden_dim = 512
n_layers = 3
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# loss and optimization functions
lr=0.001

#criterion = nn.BCELoss()
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params

num_epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net = net.to(device)



logger.info("start training. ")

best_accuracy = 0.0

#net.train()
# train for some number of epochs
#print("start training")
for epoch in range(num_epochs):
    # initialize hidden state
    h = net.init_hidden(i_batch_size)
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
        if (i) % 100 == 0:
            
            logger.info("i = %d ", i)

            print ('Epoch [{}/{}], Step {}, Loss: {:.6f}'
                   .format(epoch+1, num_epochs, i+1, loss.item()))

print("start predicting")
batch_size = i_batch_size                   
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
batch_size = i_batch_size                   
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
