import torch


import time

import logging
import logging.config


UNIT = "word" # unit of tokenization (char, word)
#UNIT = "char" # unit of tokenization (char, word)

#BATCH_SIZE = 128
#BATCH_SIZE = 16
BATCH_SIZE = 32
#BATCH_SIZE = 64

EMBED = ["char-cnn", "lookup"] # embeddings (char-cnn, lookup)
EMBED_SIZE = 300
NUM_FEATURE_MAPS = 100 # feature maps generated by each kernel
KERNEL_SIZES = [2, 3, 4]
DROPOUT = 0.5
LEARNING_RATE = 1e-4
VERBOSE = False
EVAL_EVERY = 2
SAVE_EVERY = 10

PAD = "<PAD>" # padding
SOS = "<SOS>" # start of sequence
EOS = "<EOS>" # end of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

torch.manual_seed(1)
CUDA = torch.cuda.is_available()
#CUDA = False

KEEP_IDX = False # use the existing indices when preparing additional data
NUM_DIGITS = 4 # number of digits to print


#b_partial = True
b_partial = False

b_do_colab = True
#b_do_colab = False

#b_use_new_data_set = False
b_use_new_data_set = True

#b_use_new_data_IMDB_or_YELP = True
b_use_new_data_IMDB_or_YELP = False

i_partial_count = 200
#i_partial_count = 1000
#i_partial_count = 20000
#i_partial_count = 100000

i_len_picked_words_len = 100000
#i_len_picked_words_len = 50000
#i_len_picked_words_len = 2000


s_log_config_fn = "logging.conf"


if b_do_colab:
    s_log_config_fn = "/content/drive/My Drive/gcolab/comp511prj4/cnn_code/" + s_log_config_fn

print("\n s_log_config_fn = ", s_log_config_fn)

logging.config.fileConfig(s_log_config_fn)





# create logger
logger = logging.getLogger('Project4Group36')



logger.info("program begins. ")



