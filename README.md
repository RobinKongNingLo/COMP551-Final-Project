COMP 511, Project 4, Group 36:



The  IMDB  dataset  with  10  labels  were  suggested  in  [20],  can  be  downloaded  from  the  following  link:https://www.dropbox.com/s/0oea49j7j30y671/data.json?dl=0 See https://github.com/nihalb/JMARS for details.

The Yelp 2015 dataset with 5 labels were suggested in [8],  can be downloaded from the following link:http://goo.gl/JyCnZq See https://github.com/zhangxiangxiao/Crepe for details.


To train Yelp dataset, run LSTM_yelp.py. To train IMDB dataset, run LSTM_IMDB.py. 
embedding_dim: dimension of embedding
hidden_dim: dimensiomn of hidden states in LSTM layer
n_layers: number of hidden layers


./CNN-word/ folder contains the pytorch code for CNN-word model, forked and modified based on the implementation in https://github.com/threelittlemonkeys/cnn-text-classification-pytorch

to generate the word embeding dictionary, run python prepare.py training_data 
to do the training and prediction, run python train.py model char_to_idx word_to_idx tag_to_idx training_data.csv validation_data.json 5




text_classfication_logistic_regression.py -- the code for logistic regression with different grams, using sklearn. Has bow (bag of words) and tfidf support. 

text_classfication_svm.py -- the code for svm with different grams, using sklearn. Has bow (bag of words) and tfidf support.

run group_execution_svm_orig.py which will call text_classfication_svm.py with different experiment setttings.

run group_execution_lr_v2.py or group_execution_lr.py which will call text_classfication_logistic_regression.py with different experiment settings. 


lstm-final.py -- our LSTM implementation based on pytorch. run it on kaggle kernel, for it requires GPU support. It could achieve 65% accuracy for the yelp dataset.
