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
    rating = [int(data_point["rating"])-1]
    ratings.append(rating)


print ('Number of reviews :', len(reviews))
print(reviews[15])
print(ratings[0:15])
#print(ratings.size())
ratings = np.array(ratings, dtype=int)

#ratings = ratings[:int(ratings.size(0)*1)]
all_words = ' '.join(reviews)
# create a list of words
word_list = all_words.split()
# Count all the words using Counter Method
count_words = Counter(word_list)
len_words = len(word_list)
sorted_words = count_words.most_common(len_words)
# Code words into numbers
word_to_num = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
reviews_num = []
for review in reviews:
    num = [word_to_num[w] for w in review.split()]
    reviews_num.append(num)
print (reviews_num[15])

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

reviews_pad = torch.tensor(reviews_pad)
#reviews_pad = reviews_pad[:int(reviews_pad.size(0) * 1)])
reviews_pad = torch.unsqueeze(reviews_pad, dim=1).type(torch.FloatTensor)
reviews_pad = torch.unsqueeze(reviews_pad, dim=3).type(torch.FloatTensor)
print(reviews_pad.size())
# Device configuration
print(reviews_pad[0:3])
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 64
learning_rate = 0.001
train_size = 0.9
num_workers=1
pin_memory=True
ratings = torch.tensor(ratings)
ratings = torch.squeeze(ratings, dim=1)
print(ratings.size())
torch_dataset = Data.TensorDataset(reviews_pad, ratings)
print(len(torch_dataset))
train_dataset, test_dataset = Data.random_split(torch_dataset, [int(len(torch_dataset) * train_size), len(torch_dataset) - int(len(torch_dataset) * train_size)])
print(len(train_dataset), ' ', len(test_dataset))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, 
                                           num_workers=num_workers, 
                                           pin_memory=pin_memory,)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          num_workers=num_workers, 
                                          pin_memory=pin_memory,)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=2),)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=2),)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=5),)
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=5),)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5,1), stride=5),)
        self.linear1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 1024*2, out_features = 512),)
        self.linear2 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(512, 10),)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1) 
        x = self.linear1(x)
        x = self.linear2(x)
        return x

model = ConvNet(num_classes).cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #images = torch.tensor(images)
        images = images.to(device)
        #images = images.cuda()
        #labels = torch.tensor(labels)
        labels = labels.to(device)
        #labels = labels.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted[15])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
