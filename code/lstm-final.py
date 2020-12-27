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


import time

import logging
import logging.config

import psutil

#print(psutil.cpu_percent())
#print(psutil.virtual_memory())  # physical memory usage
#print('memory % used:', psutil.virtual_memory()[2])

import gc

#b_partial = True
b_partial = False

i_partial_count = 1000

#b_do_kaggle = False
b_do_kaggle = True

#b_do_colab = True
b_do_colab = False



b_use_new_data_set = True

#b_use_new_data_IMDB_or_YELP = True
b_use_new_data_IMDB_or_YELP = False


i_batch_size = 200

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
    
    
    

logging.config.fileConfig(s_log_config_fn)


logger = logging.getLogger('Project4Group36')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")


print(psutil.virtual_memory())  # physical memory usage
print('memory % used before load dataset:', psutil.virtual_memory()[2])


str_input_folder = "../../../"

if b_do_kaggle:
    str_input_folder = "../input/yelp-imdb-multi-class-v2/datasets/"

if b_do_colab:
    str_input_folder = "/content/drive/My Drive/gcolab/dataset/datasets/"



str_json_fn_training = ""
str_json_fn_testing = ""

if b_use_new_data_set:
    
    if b_use_new_data_IMDB_or_YELP:
        str_json_fn_training = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
    else:   
        str_json_fn_training = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"
        str_json_fn_testing = "../input/yelp-test-set/data_test/data_test.json"

print(str_json_fn_training)
print(str_json_fn_testing)

data = None

data_test = None



with open(str_json_fn_training) as fp:
    data = json.load(fp)
    
i_len_of_training = len(data)

if str_json_fn_testing != "":
    
    with open(str_json_fn_testing) as fp:
        data_test = json.load(fp)
else:
    i_split_pos = int(i_len_of_training * 0.8)
    
    data_test = data[i_split_pos:]
    data = data[0:i_split_pos]
    

print("len of data_test: ", len(data_test))

if b_partial:
    data = data[0:i_partial_count]
    data_test = data_test[0:int(i_partial_count * 0.2)]


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after load dataset:', psutil.virtual_memory()[2])


#
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



reviews_test = []
ratings_fs_test = []

for data_point in data_test:
    review = data_point["review"]
    review = review.lower()
    #remove punctuation
    review = re.sub(r'[^\w\s]', ' ', review)
    reviews_test.append(review)
    rating = int(data_point["rating"])-1
    ratings_fs_test.append(rating)


print ('Number of reviews :', len(reviews_test))
print(reviews_test[15])
print(ratings_fs_test[0:15])
#print(ratings.size())
ratings_test = np.array(ratings_fs_test, dtype=int)

#all_words = ' '.join(reviews)





del ratings_fs
del ratings_fs_test

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




i_max_pick_size = 100000

i_max_pick_size = len_counted_words


print("i_max_pick_size = ", i_max_pick_size)
sorted_words = count_words.most_common(i_max_pick_size)

sorted_word_list = [w for i, (w,c) in enumerate(sorted_words)]

set_sorted_word = set(sorted_word_list)


print("len of set_sorted_word = ", len(set_sorted_word))

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


reviews_num_test = []
for review in reviews_test:
    #num = [word_to_num[w] for w in review.split()]
    num = []
    
    for w in review.split():
        if w in set_sorted_word:
            num.append(word_to_num[w])
    
    reviews_num_test.append(num)
print (reviews_num_test[0:3])

#reviews_len = [len(x) for x in reviews_num]




logger.info("after data pre-process. ")


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after data pre-process:', psutil.virtual_memory()[2])

i_word_to_num_len = len(word_to_num)


del count_words
del sorted_words
del sorted_word_list
del set_sorted_word
del reviews

del reviews_test

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



reviews_pad_test = np.zeros((len(reviews_num_test), i_len_according_to_hist), dtype = int)

for i, review in enumerate(reviews_num_test):
    review_len = len(review)

    if review_len <= i_len_according_to_hist:
        zeroes = list(np.zeros(i_len_according_to_hist - review_len))
        new = review + zeroes
    elif review_len > i_len_according_to_hist:
        new = review[0:i_len_according_to_hist]

    reviews_pad_test[i,:] = np.array(new)



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

#val_test_x = reviews_pad[int(train_size*len(reviews_pad)):]
#val_test_y = ratings[int(train_size*len(reviews_pad)):]


val_x = reviews_pad[int(train_size*len(reviews_pad)):]
val_y = ratings[int(train_size*len(reviews_pad)):]

#val_x = val_test_x[0:int(len(val_test_x)*0.5)]
#val_y = val_test_y[0:int(len(val_test_y)*0.5)]

test_x = reviews_pad_test[:]
test_y = ratings_test[:]


print(psutil.virtual_memory())  # physical memory usage
print('memory % used after data prepare:', psutil.virtual_memory()[2])


del reviews_num
del reviews_pad
del ratings

del reviews_num_test
del reviews_pad_test
del ratings_test

#del val_test_x
#del val_test_y


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
print(train_y_t.size())

print(val_x_t.size())
print(val_y_t.size())

print(test_x_t.size())
print(test_y_t.size())


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


g_batch_size = i_batch_size

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=i_batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=i_batch_size)



#valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=i_batch_size)








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
        self.batch_size = g_batch_size
        
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
        
        #print("out after self.fc:", out.shape, out)
        #print("\n item 1:", out[1].shape, out[1])
        out = F.softmax(out)
        
        #out_lsed = F.log_softmax(out)
        #out_lsed = out
        #print("out after softmax:", out.shape, out)
        #print("\n item 1:", out[1].shape, out[1])
        
        # reshape to be batch_size first
        #print("g_batch_size = ", g_batch_size)
        #print("self.batch_size = ", self.batch_size)
        #print("out before out.view:", out.shape, out.shape[0])
        
        i_real_batch_size = out.shape[0] / i_len_according_to_hist
        i_real_batch_size = int(i_real_batch_size)
        #print("i_real_batch_size = ", i_real_batch_size)
        
        out = out.view(i_real_batch_size, -1, i_output_size)
        #print("out after out.view:", out.shape)
        
        
        #print("\n item 1:", out[1].shape, out[1])
        
        out = out[:, -1] # get last batch of labels
        
        
        
        #print("out after out[:, -1]:", out.shape, out)
        # return last sigmoid output and hidden state
        #print("\n item 1:", out[1].shape, out[1])
        
        return out, hidden
        #return out, hidden, out_lsed
    
    
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
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# loss and optimization functions
lr=0.001

#criterion = nn.BCELoss()

#criterion = nn.MSELoss(size_average=False)
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# training params

num_epochs = 20 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
if(train_on_gpu):
    net = net.to(device)



logger.info("start training. ")

best_accuracy = 0.0
model_parameter = net.state_dict()

#net.train()
# train for some number of epochs
#print("start training")
for epoch in range(num_epochs):
    # initialize hidden state
    h = net.init_hidden(i_batch_size)
    # batch loop
    net.train()
    for i, (inputs, labels) in enumerate(train_loader):

        if(train_on_gpu):
            inputs, labels = inputs.to(device), labels.to(device)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        #h = net.init_hidden()
        g_batch_size = inputs.size(0)
        
        #print("inputs shape", inputs.shape)
        #print("g_batch_size = ", g_batch_size)
        
        h = net.init_hidden(batch_size = g_batch_size)
        
        #h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.cuda.LongTensor)
        output, h = net(inputs, h)

        #print("shape of output", output.shape)
        #print("shape of labels", labels.shape)

        # calculate the loss and perform backprop
        #labels = labels.unsqueeze(0)
        #loss = criterion(output, labels.float())
        
        #pred_y = torch.max(output, 1)[1].data.numpy()
        
        loss = criterion(output, labels)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()
        if (i) % 100 == 0:
            
            logger.info("i = %d ", i)

            print ('Epoch [{}/{}], Step {}, Loss: {:.6f}'
                   .format(epoch+1, num_epochs, i+1, loss.item()))



    logger.info("validating aftre epoch %d: ", epoch)
    #batch_size = i_batch_size                   
    val_h = net.init_hidden(i_batch_size)
    right_num = list()
    total_num = list()
    
    net.eval()
    for i, (vali_x, vali_y) in enumerate(valid_loader):
        vali_x = vali_x.to(device)
        g_batch_size = vali_x.size(0)
        val_h = net.init_hidden(batch_size = g_batch_size)
        
        #print("vali_x shape", vali_x.shape)
        #print("g_batch_size = ", g_batch_size)
        
        net.zero_grad()
        preds_y, val_h = net(vali_x, val_h)
        
        """
        print("len of pred_y = ", len(pred_y), pred_y.shape)
        print(pred_y[1])
        """
        
        """
        pred_y = pred_y.cpu().data.numpy()
        #print(pred_y[1])
        pred_y = np.rint(pred_y)
        #print(pred_y[1])
        """
        
        pred_y = torch.max(preds_y, 1)[1].cpu().data.numpy()
        
        #pred_y = torch.max(vali_output, 1)[1].cpu().data.numpy()
        
        #print("len of pred_y_lsed = ", len(pred_y_lsed), pred_y_lsed.shape)
        #print(pred_y_lsed[1])
        
                # print(vali_output.size())
        #print(vali_y.size())
        #_, pred_y = torch.max(pred_y.data, 0)
                # print(pred_y.shape)
                # print(float((pred_y == vali_y.numpy()).sum()), len(pred_y))
                # accuracy = float((pred_y == vali_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        right_num.append(float((pred_y == vali_y.numpy()).sum()))
        #right_num.append(int(torch.sum(tf.equal(pred_y == vali_y))))
        total_num.append(len(pred_y))
        if (i+1) % 10 == 0:
            #print ('val step', i+1)
            pass
    accuracy = sum(right_num) / sum(total_num)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model_parameter = net.state_dict()
    
    #logger.info("report vali accuracy: ")
    print('vali accuracy: %.6f' % accuracy, 'Total right: ', sum(right_num), '| Total: ', sum(total_num))
    

torch.save(model_parameter, 'model_para.pkl')

logger.info("best_accuracy: %f", best_accuracy)
    

def prediction_print(y_test, y_test_probs):
    
    
    cur_time = int(time.time())
    
    str_postfix = ""
    if b_use_new_data_IMDB_or_YELP:
        str_postfix = "imdb"
    else:
        str_postfix = "yelp"
    
    csv_fn = "prediction_lstm_" + str(cur_time) + "_" + str_postfix + ".csv"
    
    print("\n csv_fn = ", csv_fn)
    
    with open(csv_fn, 'wt', newline='') as csv_file:
        
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        header = ['Id', 'Category']
    
        for idx in range(0, i_output_size, 1):
            header.append("l_" + str(idx + 1))
        
        writer.writerow(header)        
        
        count = 0
        for data_point in y_test_probs:
            
                        
            row = [count, (y_test[count] + 1)]
            for idx in range(0, i_output_size, 1):
                #str_title = "l_" + str(idx + 1)
                row.append(data_point[idx])
        

            writer.writerow(row)

            count += 1

    csv_file.close()


def test_model(model_para, device, ):
    #cnn_test = CNNv9().to(device)
    
    g_batch_size = i_batch_size
    print()
    
    net_test = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers).to(device)
    net_test.load_state_dict(model_para)
    print('a')
    
    """
    test_images = None
    if b_dl_new_or_old:
    
        test = pd.read_csv(input_data_folder + str_test_file)
    
        test_images = test.iloc[:, 0:]
        
        test_images = test_images.values
        
        test_images = np.reshape(test_images, (test_images.shape[0], 64, 64))
        
    else:
        test_images = pd.read_pickle(str_folder + 'test_images.pkl')    
    
    
    print("test_images debug:", type(test_images), test_images.shape, test_images[2])
    
    X = torch.tensor(test_images / 255.)
    X = torch.unsqueeze(X, dim=1).type(torch.FloatTensor)
    pred_y = list()
    """
    test_accuracy = 0.0
    
    net_test.eval()
    
    """
    for x in X:
        x = x.view(1, 1, 64, 64)
        x = x.to(device)        
        test_output = net_test(x)
        pred_y.append(torch.max(test_output, 1)[1].cpu().data.numpy()[0])
    """
        
    
    pred_y_list = list()
    pred_y_prob_list = list()
    
    #batch_size = i_batch_size                   
    test_h = net_test.init_hidden(i_batch_size)
    right_num = list()
    total_num = list()
    for i, (test_x, test_y) in enumerate(test_loader):
        test_x = test_x.to(device)
        g_batch_size = test_x.size(0)
        
        #print("test_x shape", test_x.shape)
        #print("g_batch_size = ", g_batch_size)
        
        test_h = net_test.init_hidden(batch_size = g_batch_size)
        #net_test.zero_grad()
        preds_y, test_h = net_test(test_x, test_h)
        
        """
        print("len of pred_y = ", len(pred_y), pred_y.shape)
        print(pred_y[1])
        """
        
        """
        pred_y = pred_y.cpu().data.numpy()
        #print(pred_y[1])
        pred_y = np.rint(pred_y)
        #print(pred_y[1])
        """
        
        pred_y = torch.max(preds_y, 1)[1].cpu().data.numpy()
        
        pred_y_list.extend(pred_y.tolist())
        
        pred_y_prob_list.extend(preds_y.cpu().data.numpy().tolist())
        
        #pred_y = torch.max(vali_output, 1)[1].cpu().data.numpy()
        
        #print("len of pred_y_lsed = ", len(pred_y_lsed), pred_y_lsed.shape)
        #print(pred_y_lsed[1])
        
                # print(vali_output.size())
        #print(vali_y.size())
        #_, pred_y = torch.max(pred_y.data, 0)
                # print(pred_y.shape)
                # print(float((pred_y == vali_y.numpy()).sum()), len(pred_y))
                # accuracy = float((pred_y == vali_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        right_num.append(float((pred_y == test_y.numpy()).sum()))
        #right_num.append(int(torch.sum(tf.equal(pred_y == vali_y))))
        total_num.append(len(pred_y))
        if (i+1) % 100 == 0:
            print ('test step', i+1)
            #pass
            
    accuracy = sum(right_num) / sum(total_num)
    
    
    if accuracy > test_accuracy:
        test_accuracy = accuracy
        
    
    #logger.info("report vali accuracy: ")
    print('test accuracy: %.6f' % accuracy, 'Total right: ', sum(right_num), '| Total: ', sum(total_num)) 
    
    print("len of pred_y_list: ", len(pred_y_list))
    print("len of pred_y_prob_list: ", len(pred_y_prob_list))
    prediction_print(pred_y_list, pred_y_prob_list)


test_model(model_parameter, device)






"""
if(train_on_gpu):
    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.cuda.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                inputs = inputs.type(torch.LongTensor)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
"""