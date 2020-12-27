#!/usr/bin/env python3



import os

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer


from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_predict

import numpy as np

from sklearn.exceptions import FitFailedWarning
import warnings

from sklearn.pipeline import Pipeline


from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import f1_score

import csv



import time

import logging
import logging.config

import json

import sys

#from os import listdir
#from os.path import isfile, join

#from pathlib import Path

#import re

#from bs4 import BeautifulSoup


#print(os.listdir("../input/yelp-imdb-multi-class/datasets/datasets/"))



#'imdb_or_yelp', 'tfidf_or_bow', 'i_max_feature_size', 'i_gram'

#
str_arg_1 = str(sys.argv[1])

str_arg_2 = str(sys.argv[2])

i_arg_3 = int(sys.argv[3])

i_arg_4 = int(sys.argv[4])








b_do_kaggle = False
#b_do_kaggle = True






i_cross_validation_fold_count = 5

#b_partial = True
b_partial = False



b_cleanup = True


b_wo_sw = False

b_w_lemmatize = True

b_w_stemming = False


b_w_nwn = True

b_wo_punctuation = False



b_sentiment_words_filter = True


b_negate = b_sentiment_words_filter


#b_negate = True

b_separate_wp = True



i_partial_count = 1000


b_do_model_selection = False
#b_do_model_selection = True

b_do_cross_validation = False
#b_do_cross_validation = True


#b_use_new_data_set = False
b_use_new_data_set = True



#b_use_new_data_IMDB_or_YELP = True
b_use_new_data_IMDB_or_YELP = False

if str_arg_1 == "imdb":
    b_use_new_data_IMDB_or_YELP = True
elif str_arg_1 == "yelp":
    b_use_new_data_IMDB_or_YELP = False
else:
    pass



s_log_config_fn = "logging.conf"


if b_do_kaggle:
    s_log_config_fn = "../input/config/" + s_log_config_fn

logging.config.fileConfig(s_log_config_fn)





# create logger
logger = logging.getLogger('Project2Group12')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")


str_fn_postfix = ""


#b_Tfidf_or_BOW = False
b_Tfidf_or_BOW = True

if str_arg_2 == "tfidf":
    b_Tfidf_or_BOW = True
elif str_arg_2 == "bow":
    b_Tfidf_or_BOW = False
else:
    pass

i_ngram_range_upbound = 2

i_ngram_range_upbound = i_arg_4

#i_max_features_cnt = 50000
i_max_features_cnt = 500000

i_max_features_cnt = i_arg_3


print("b_Tfidf_or_BOW = ", b_Tfidf_or_BOW, "\n")

print("i_ngram_range_upbound = ", i_ngram_range_upbound, "\n")

print("b_use_new_data_IMDB_or_YELP = ", b_use_new_data_IMDB_or_YELP, "\n")



print("i_max_features_cnt = ", i_max_features_cnt, "\n")


i_n_cv_jobs = 1

if i_ngram_range_upbound <= 2:
    i_n_cv_jobs = 3
elif i_ngram_range_upbound > 2 and i_ngram_range_upbound <= 3:
    i_n_cv_jobs = 2
else:
    #i_ngram_range_upbound = 5
    i_n_cv_jobs = 1

print("i_n_cv_jobs = ", i_n_cv_jobs, "\n")

if b_cleanup:
    str_fn_postfix += "_cleanup"


if b_wo_sw:
    str_fn_postfix += "_wo_sw"

if b_negate:
    str_fn_postfix += "_negate"

if b_wo_punctuation:
    str_fn_postfix += "_wo_punctuation"

if b_w_lemmatize:
    str_fn_postfix += "_w_lemmatize"
    

if b_w_stemming:
    str_fn_postfix += "_w_stemming"


if b_w_nwn:
    str_fn_postfix += "_w_nwn"
    
    
if b_sentiment_words_filter:
    str_fn_postfix += "_w_swf"

if b_separate_wp:
    str_fn_postfix += "_w_separate_wp"



if True:
    str_fn_postfix += "_stat"



str_fn_postfix += "_simplified"


b_use_original_text = True


#str_fn_postfix = "_cleanup_negate_w_lemmatize_w_nwn_w_swf_w_separate_wp_stat_simplified"




str_json_fn_training = "../" + "training" + "_set" + str_fn_postfix + ".json"


str_input_folder = "../../../"

if b_do_kaggle:
    str_input_folder = "../input/yelp-imdb-multi-class/datasets/"

if b_use_new_data_set:
    
    if b_use_new_data_IMDB_or_YELP:
        str_json_fn_training = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
    else:   
        str_json_fn_training = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"
    


str_json_fn_testing = "../" + "testing" + "_set" + str_fn_postfix + ".json"

if b_use_new_data_set:
    
    if b_use_new_data_IMDB_or_YELP:
        str_json_fn_testing = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
    else:   
        str_json_fn_testing = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data_test.json"


logger.info("str_json_fn_training = %s \n", str_json_fn_training)

logger.info("str_json_fn_testing = %s \n", str_json_fn_testing)

logger.info("begin pre-process for training datas... ")

X_list_raw = []
Y_list_raw = []

all_files_list_raw = []



jsonFile = open(str_json_fn_training, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file


real_training_data_set = []

if not b_partial:
    
    if b_use_new_data_set:
        real_training_data_set = [item for item in json_data]
    else:    
        real_training_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    if b_use_new_data_set:
        real_training_data_set = [item for item in json_data]
        real_training_data_set = [item for item in real_training_data_set[:i_partial_count]]
    else:
        real_training_data_set = sorted(json_data, key=lambda x: (x['id']), reverse=False)    
        real_training_data_set = [item for item in real_training_data_set[:i_partial_count] if item['y_val'] > -2]
    
    
#real_training_data_set_sorted = sorted(real_training_data_set, key=lambda x: (x['id']), reverse=False)
real_training_data_set_sorted = real_training_data_set

logger.info("len of real_training_data_set_sorted = %d ", len(real_training_data_set_sorted))

tmp_list_X_raw = []
tmp_list_Y_raw = []
#tmp_files_list_raw = []
#tmp_pure_files_list_raw = []


for data_point in real_training_data_set_sorted:
    

    i_se = 0
    #i_se = data_point['i_sentiment_estimate']
    
    str_st = ""
    
    if b_use_original_text:
        #str_st = data_point['text']
        #str_st = data_point['text_simple_cleanup']
        
        #str_st = data_point['raw_text']
        #str_st = data_point['simplified_cleanup']        
        
        
        #str_st = data_point['str_nwn']
        #str_st = data_point['text_letter_and_num']
        
        #str_st = data_point['text_wo_html_tags']
        
        if b_use_new_data_set:
            str_st = data_point['review']
        else:
            str_st = data_point['simplified_cleanup']
        
        
    else:
        str_st = data_point['senti_text']
    
        if i_se >= 15:
            str_st += " . imdbsuperpositive "
        elif i_se < 15 and i_se >= 5:
            str_st += " . imdbstrongpositive "
        elif i_se < 5 and i_se > -5:
            pass
        elif i_se <= -5 and i_se > -15:
            str_st += " . imdbstrongnegitive "
        elif i_se <= -15:
            str_st += " . imdbsupernegitive "
        else:
            pass
    
    tmp_list_X_raw.append(str_st)
    if b_use_new_data_set:
        tmp_list_Y_raw.append(data_point['rating'])
        
    else:
        tmp_list_Y_raw.append(data_point['y_val'])
    
    #tmp_files_list_raw.append(data_point['full_file_name'])    
    #tmp_pure_files_list_raw.append(data_point['id'])

X_list_raw.extend(tmp_list_X_raw)
Y_list_raw.extend(tmp_list_Y_raw)

#all_files_list_raw.extend(tmp_files_list_raw)


logger.info("end pre-process for training datas... ")





logger.info("begin pre-process for testing datas... ")


X_list_real_test_raw = []
Y_list_real_test_raw = []

#all_files_list_real_test_raw = []
#all_pure_files_list_real_test_raw = []



jsonFile = open(str_json_fn_testing, "r") # Open the JSON file for reading
json_data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close() # Close the JSON file


real_testing_data_set = []

"""
if not b_partial:
    real_testing_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    real_testing_data_set = [item for item in json_data[:i_partial_count] if item['y_val'] > -2]


real_testing_data_set_sorted = sorted(real_testing_data_set, key=lambda x: (x['id']), reverse=False)
"""



if not b_partial:
    
    if b_use_new_data_set:
        real_testing_data_set = [item for item in json_data]
    else:    
        real_testing_data_set = [item for item in json_data if item['y_val'] > -2]
else:
    if b_use_new_data_set:
        real_testing_data_set = [item for item in json_data]
        real_testing_data_set = [item for item in real_testing_data_set[:i_partial_count]]
    else:
        real_testing_data_set = sorted(json_data, key=lambda x: (x['id']), reverse=False)    
        real_testing_data_set = [item for item in real_testing_data_set[:i_partial_count] if item['y_val'] > -2]



real_testing_data_set_sorted = real_testing_data_set



logger.info("len of real_testing_data_set_sorted = %d ", len(real_testing_data_set_sorted))

tmp_list_X_raw = []
tmp_list_Y_raw = []
tmp_files_list_raw = []
tmp_pure_files_list_raw = []


for data_point in real_testing_data_set_sorted:
    
    #tmp_list_X_raw.append(data_point['senti_text'])
    
    i_se = 0
    #i_se = data_point['i_sentiment_estimate']
    
    str_st = ""   
    if b_use_original_text:
        #str_st = data_point['text']
        #str_st = data_point['text_simple_cleanup']
        
        
        #str_st = data_point['str_nwn']
        
        
        #str_st = data_point['raw_text']
        #str_st = data_point['text_letter_and_num']
        #str_st = data_point['simplified_cleanup']
        #str_st = data_point['text_wo_html_tags']
        
        if b_use_new_data_set:
            str_st = data_point['review']
        else:
            str_st = data_point['simplified_cleanup']
        
    
    else:
        str_st = data_point['senti_text']
    
        if i_se >= 15:
            str_st += " . imdbsuperpositive "
        elif i_se < 15 and i_se >= 5:
            str_st += " . imdbstrongpositive "
        elif i_se < 5 and i_se > -5:
            pass
        elif i_se <= -5 and i_se > -15:
            str_st += " . imdbstrongnegitive "
        elif i_se <= -15:
            str_st += " . imdbsupernegitive "
        else:
            pass

    tmp_list_X_raw.append(str_st)

    #tmp_list_Y_raw.append(data_point['y_val'])
    
    if b_use_new_data_set:
        tmp_list_Y_raw.append(data_point['rating'])
        
    else:
        tmp_list_Y_raw.append(data_point['y_val'])
    

    #tmp_files_list_raw.append(data_point['full_file_name'])    
    #tmp_pure_files_list_raw.append(data_point['id'])

X_list_real_test_raw.extend(tmp_list_X_raw)
Y_list_real_test_raw.extend(tmp_list_Y_raw)

#all_files_list_real_test_raw.extend(tmp_files_list_raw)
#all_pure_files_list_real_test_raw.extend(tmp_pure_files_list_raw)


logger.info("end pre-process for testing datas... ")






# seems that the all data (both training and testing) are combined together and then being splited, 
# according to train_size and test_size parameters of train_test_split
#X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, train_size=0.6, test_size=0.4)


logger.info("start train_test_split. ")


i_random_state=20


print("len of X_list_raw = ", len(X_list_raw), type(X_list_raw))
print("len of Y_list_raw = ", len(Y_list_raw), type(Y_list_raw))

X_list_raw_real_test = []
Y_list_raw_real_test = []

if b_use_new_data_set and not b_use_new_data_IMDB_or_YELP:

    X_list_raw_real_test = X_list_real_test_raw[:]
    Y_list_raw_real_test = Y_list_real_test_raw[:]
    
else:
    
    

    i_train_len = int(0.8 * len(X_list_raw))
    print("i_train_len = ", i_train_len)

    X_list_raw_real_test = X_list_raw[i_train_len:]
    X_list_raw = X_list_raw[0:i_train_len]

    Y_list_raw_real_test = Y_list_raw[i_train_len:]
    Y_list_raw = Y_list_raw[0:i_train_len]
    
    
    

print("len of X_list_raw_real_test = ", len(X_list_raw_real_test))
print("len of Y_list_raw_real_test = ", len(Y_list_raw_real_test))


#X_train, X_test, Y_train, Y_test, all_files_train, all_files_test = train_test_split(X_list_raw, Y_list_raw, all_files_list_raw, train_size=0.8, test_size=0.2, random_state=i_random_state)
X_train, X_test, Y_train, Y_test = train_test_split(X_list_raw, Y_list_raw, train_size=0.8, test_size=0.2, random_state=i_random_state)


logger.info("end train_test_split. ")

print("len of X_train, X_test, Y_train, Y_test \n", len(X_train), len(X_test), len(Y_train), len(Y_test))

i_random_index = 15

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_train: ==============================================================\n")
print(X_train[i_random_index])

print("\n\n\n\n item of Y_train: ==============================================================\n")
print(Y_train[i_random_index])

#print("\n\n\n\n item of all_files_train: ==============================================================\n")
#print(all_files_train[i_random_index])




i_random_index = 18

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_test: ==============================================================\n")
print(X_test[i_random_index])

print("\n\n\n\n item of Y_test: ==============================================================\n")
print(Y_test[i_random_index])

#print("\n\n\n\n item of all_files_test: ==============================================================\n")
#print(all_files_test[i_random_index])





i_random_index = 20

print("\n\n i_random_index = ", i_random_index)

print("\n\n\n\n item of X_list_real_test_raw: ==============================================================\n")
print(X_list_real_test_raw[i_random_index])

print("\n\n\n\n item of Y_list_real_test_raw: ==============================================================\n")
print(Y_list_real_test_raw[i_random_index])

#print("\n\n\n\n item of all_files_list_real_test_raw: ==============================================================\n")
#print(all_files_list_real_test_raw[i_random_index])





X_train_whole = []

X_train_whole.extend(X_train)
X_train_whole.extend(X_test)

Y_train_whole = []

Y_train_whole.extend(Y_train)
Y_train_whole.extend(Y_test)


#all_files_whole_train = []

#all_files_whole_train.extend(all_files_train) 
#all_files_whole_train.extend(all_files_test)






print("len of X_train_whole and Y_train_whole", len(X_train_whole), len(Y_train_whole))


print("len of X_train and Y_train", len(X_train), len(Y_train))




logger.info("start CountVectorizer. ")

vect = CountVectorizer(min_df=0.0002, max_df=0.95, max_features=i_max_features_cnt, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, i_ngram_range_upbound))

count_vect = vect.fit(X_train_whole)

logger.info("after vect.fit. ")

X_train_counts = count_vect.transform(X_train)
X_test_counts = count_vect.transform(X_test)

X_real_test_counts = count_vect.transform(X_list_raw_real_test)


logger.info("end CountVectorizer. ")




logger.info("start TfidfTransformer. ")

tfidf = TfidfTransformer(use_idf=b_Tfidf_or_BOW, smooth_idf=b_Tfidf_or_BOW, sublinear_tf=b_Tfidf_or_BOW)

tfidf_transformer = tfidf.fit(X_train_counts)

X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

X_real_test_tfidf = tfidf_transformer.transform(X_real_test_counts)


logger.info("end TfidfTransformer. ")



logger.info("start Normalizer. ")

norm = Normalizer()

normalizer_tranformer = norm.fit(X_train_tfidf)
X_train_normalized = normalizer_tranformer.transform(X_train_tfidf)
X_test_normalized = normalizer_tranformer.transform(X_test_tfidf)

X_real_test_normalized = normalizer_tranformer.transform(X_real_test_tfidf)


logger.info("end Normalizer. ")






"""
pclf = Pipeline([
    #('vect', CountVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4), stop_words = 'english')),
    #('vect', CountVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    #('vect', CountVectorizer(min_df=0.0002, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    #('vect', CountVectorizer(min_df=0.0002, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 4))),
    ('vect', CountVectorizer(min_df=0.0002, max_df=1.0, max_features=i_max_features_cnt, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, i_ngram_range_upbound))),
    #('vect', CountVectorizer(min_df=0.0, max_df=1.0, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1))),
    #('tfidf', TfidfTransformer(use_idf=False, smooth_idf=False, sublinear_tf=False)),
    ('tfidf', TfidfTransformer(use_idf=b_Tfidf_or_BOW, smooth_idf=b_Tfidf_or_BOW, sublinear_tf=b_Tfidf_or_BOW)),
    ('norm', Normalizer()),
    #('clf', LogisticRegression(penalty = 'l2', dual = False, random_state = 0, solver='sag', C=20, max_iter=1000, multi_class='multinomial')),
    ('clf', LogisticRegression(penalty = 'l2', dual = True, random_state = 0, solver='liblinear', C=20, max_iter=1000, multi_class='auto')),
    #('clf', LogisticRegression(penalty = 'l2', dual = True, random_state = 0, solver='liblinear', C=20, max_iter=1000, multi_class='auto')),
])
"""    

clf = LogisticRegression(penalty = 'l2', dual = True, random_state = 0, solver='liblinear', C=20, max_iter=1000, multi_class='auto')
    
    
#pclf.fit(X_train_whole, Y_train_whole)

#pclf.fit(X_train, Y_train)
    
    

#params = {"C":[1, 4, 16, 32]
#}

    
""" 
params = {"vect__ngram_range": [(1,4)],
             "vect__min_df" : [2], 
             "vect_max_df" : [0.5, 0.6, 0.7, 0.8],
            "clf__C": [20]
}
"""

"""
params = { 
          "vect_max_df" : [0.5, 0.6, 0.7, 0.8]

}
"""


""" 
params = {"vect__ngram_range": [(1,2),(1,3),(1,4)],
            "clf__C": [1, 30]
}
"""
    
"""
params = {"vect__ngram_range": [(1,1),(1,2),(1,3),(1,4)],
            "clf__C": [1, 15, 30]
}
"""
    
"""
params = {"clf__C": [1, 4, 16, 32]
}
"""

params = {"clf__C": [20]
}


model_search_LR = GridSearchCV(clf, param_grid = params, scoring = 'roc_auc', cv = 5, verbose = 10, error_score='raise', iid=True, n_jobs=2)




st = int(time.time())

logger.info("start fit for pclf")

if b_do_model_selection:

    with warnings.catch_warnings(record=True) as w:
        try:
            #gs.fit(X, y)   # This will raise a ValueError since C is < 0        
            model_search_LR.fit(X_train_whole, Y_train_whole) # Fit the model.
            #model_search_LR.fit(X_train, Y_train) # Fit the model.
        except Exception as e:
            #except ValueError:
            print("\n\n Exception e = " + str(e))
            #pass
        print("\n\n w = \n", w)

    # Try to set the scoring on what the contest is asking for. 
    # The contest says scoring is for area under the ROC curve, so use this.
    
    #error_score=np.nan
    
    
    print("\n\n model_search_LR.scorer_ = \n", model_search_LR.scorer_) 
    
    print("\n\n model_search_LR.best_index_ = \n", model_search_LR.best_index_) 
    
    print("\n\n model_search_LR.best_score_ = \n", model_search_LR.best_score_)                        
    
    print("\n\n model_search_LR.best_params_ = \n", model_search_LR.best_params_)
    
    print("\n\n model_search_LR.best_estimator_ = \n", model_search_LR.best_estimator_)
    
    
    print("\n\n model_search_LR.cv_results_ = \n", model_search_LR.cv_results_)


else:
    
    if b_do_cross_validation:
        
        cv_results = cross_validate(clf, X_train_normalized, Y_train_whole, scoring='f1_weighted', cv=i_cross_validation_fold_count, n_jobs=i_n_cv_jobs, return_train_score=False, verbose=10)
    
        logger.info(sorted(cv_results.keys()))
    
        logger.info(cv_results)
        
        
        tmp_scores = cv_results['test_score']
        print("tmp_scores = ", type(tmp_scores), tmp_scores, tmp_scores.shape)
        
        mean_f1_from_cv = np.mean(tmp_scores)
        
        
        print("\n\n mean_f1_from_cv = ", mean_f1_from_cv)
        #logger.info("", mean_f1_from_cv)
        
        
    else:
        
        #pclf.fit(X_train_whole, Y_train_whole)
        #clf.fit(X_train, Y_train)
        clf.fit(X_train_normalized, Y_train)
    


logger.info("\n end fit for pclf")



et = int(time.time())


spend_time = et - st


logger.info("\n spend_time for the fit = %d", spend_time)


print("\n spend_time for the fit = ", spend_time)



#tmp_vect_X = pclf.named_steps['vect'].X


#print("\n\n\n\n tfidf idf_", tmp_vect_X, len(tmp_vect_X), type(tmp_vect_X), tmp_vect_X.shape)

tmp_fns = []



if b_do_model_selection:
    tmp_fns = model_search_LR.best_estimator_.named_steps['vect'].get_feature_names()
else:
    
    if b_do_cross_validation:
        tmp_fns = vect.get_feature_names()
    else:
        #tmp_fns = clf.named_steps['vect'].get_feature_names()
        tmp_fns = vect.get_feature_names()

i_len_fns = len(tmp_fns)

print("\n\n\n\n i_len_fns = ", i_len_fns, type(tmp_fns))




#list_Y_test_pred = Y_test_pred.tolist()


#print("\n\n\n\n vect get_feature_names", pclf.named_steps['vect'].get_feature_names())
"""

tmp_idfs = []

if b_do_model_selection:

    tmp_idfs = model_search_LR.best_estimator_.named_steps['tfidf'].idf_
else:
    
    tmp_idfs = pclf.named_steps['tfidf'].idf_


print("\n\n\n\n tfidf idf_", tmp_idfs, len(tmp_idfs), type(tmp_idfs), tmp_idfs.shape)

list_tmp_idfs = tmp_idfs.tolist()


print("\n\n\n\n clf list_tmp_idfs", len(list_tmp_idfs))


tmp_coefs = []

if b_do_model_selection:
    
    tmp_coefs = model_search_LR.best_estimator_.named_steps['clf'].coef_
    
else:
    
    tmp_coefs = pclf.named_steps['clf'].coef_

print("\n\n\n\n clf coef_", tmp_coefs, len(tmp_coefs), type(tmp_coefs), tmp_coefs.shape)

tmp_coefs_line = tmp_coefs[0]

list_tmp_coefs = tmp_coefs_line.tolist()


print("\n\n\n\n clf list_tmp_coefs", len(list_tmp_coefs))


tmp_intercept = []


if b_do_model_selection:
    
    tmp_intercept = model_search_LR.best_estimator_.named_steps['clf'].intercept_
    
else:
    tmp_intercept = pclf.named_steps['clf'].intercept_


print("\n\n\n\n clf intercept_", tmp_intercept, len(tmp_intercept), type(tmp_intercept))



pclf_feature_set = []

for x in range(0, i_len_fns, 1):
    
    pclf_feature_item = {}
    
    pclf_feature_item['term'] = tmp_fns[x]

    pclf_feature_item['idf'] = list_tmp_idfs[x]
    
    pclf_feature_item['coef'] = list_tmp_coefs[x]

    pclf_feature_set.append(pclf_feature_item)



pclf_feature_set_sorted = sorted(pclf_feature_set, key=lambda x: (x['idf']), reverse=False)

print("\n\n\nn\ len of pclf_feature_set_sorted by idf", len(pclf_feature_set_sorted))

#print(pclf_feature_set_sorted[0], pclf_feature_set_sorted[i_len_fns - 1])



i_buffer = 1000

print("\n\n\n\n  least: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("\n\n", pclf_feature_set_sorted[x])


print("\n\n\n\n  largest: \n", i_buffer)

for x in range(i_len_fns - 1, i_len_fns - 1 - i_buffer, -1):
    print("\n\n", pclf_feature_set_sorted[x])




pclf_feature_set_sorted = sorted(pclf_feature_set, key=lambda x: (abs(x['coef'])), reverse=False)

print("\n\n\nn\ len of pclf_feature_set_sorted by coef", len(pclf_feature_set_sorted))

#print(pclf_feature_set_sorted[0], pclf_feature_set_sorted[i_len_fns - 1])


i_buffer = 1000

print("\n\n\n\n  least: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("\n\n", pclf_feature_set_sorted[x])


print("\n\n\n\n  largest: \n", i_buffer)

for x in range(i_len_fns - 1, i_len_fns - 1 - i_buffer, -1):
    print("\n\n", pclf_feature_set_sorted[x])

"""

#print("\n\n", pclf.coef_, "\n\n")
#print("\n\n", pclf.intercept_, "\n\n")

logger.info("start predict for X_test ")

#Y_test_pred = model_search_LR.predict(X_test)

#Y_test_cv_pred = cross_val_predict(pclf, X_test, Y_test, cv=5, n_jobs=2)



st = int(time.time())

Y_test_pred = []

Y_real_test_pred = []

Y_real_test_pred_prob = []

list_Y_real_test_pred_prob = []


i_lable_cnt = 0

if b_do_model_selection:
    Y_test_pred = model_search_LR.best_estimator_.predict(X_test)
else:
    if b_do_cross_validation:
        #Y_test_pred = cross_val_predict(pclf, X_train_whole, Y_train_whole, cv=i_cross_validation_fold_count)
        pass
    else:
        Y_test_pred = clf.predict(X_test_normalized)
        Y_real_test_pred = clf.predict(X_real_test_normalized)
        
        
        Y_real_test_pred_prob = clf.predict_proba(X_real_test_normalized)
        
        list_Y_real_test_pred_prob = Y_real_test_pred_prob.tolist()
        
        
        i_lable_cnt = len(list_Y_real_test_pred_prob[0])
        
        for idx in range(2, 5, 1):
            print("\n idx = ", idx, ": ", X_list_raw_real_test[idx])
            print("\n Y_list_raw_real_test: ", Y_list_raw_real_test[idx])
            print("\n Y_real_test_pred: ", Y_real_test_pred[idx])
            print("\n list_Y_real_test_pred_prob: ", list_Y_real_test_pred_prob[idx])
            print("\n\n\n\n")
            
        
print("type and len of Y_real_test_pred_prob", type(Y_real_test_pred_prob), len(Y_real_test_pred_prob))


et = int(time.time())


spend_time = et - st

logger.info("\n spend_time for the predict = %d", spend_time)


print("\n spend_time for the predict = ", spend_time)

"""
Y_test_cv_pred = []
Y_test_cv_pred.extend([0] * int(len(Y_test) / 2))
Y_test_cv_pred.extend([1] * int(len(Y_test) / 2))

print("\n\n type and len of Y_test_cv_pred: ", type(Y_test_cv_pred), len(Y_test_cv_pred))
"""

logger.info("end predict for X_test")




print("\n\n\n\n")



logger.info("start metrics.classification_report ")

print("\n\nmetrics.classification_report for Y_test and Y_test_pred \n")


if b_do_cross_validation:
    #print(metrics.classification_report(Y_train_whole, Y_test_pred, digits=5))
    pass
else:
    print("\n\n for test:")
    print(metrics.classification_report(Y_test, Y_test_pred, digits=5))
    print("\n\n for real test:")
    print(metrics.classification_report(Y_list_raw_real_test, Y_real_test_pred, digits=5))


#print("\n\nmetrics.classification_report for Y_test and Y_test_cv_pred \n")

#print(metrics.classification_report(Y_test, Y_test_cv_pred, digits=5))

logger.info("end metrics.classification_report ")










if b_do_cross_validation:
    #quit()
    pass

"""
logger.info("start predict for X_list_real_test_raw ")

Y_real_test_pred = []

if b_do_model_selection:
    Y_real_test_pred = model_search_LR.best_estimator_.predict(X_list_real_test_raw)
else:

    Y_real_test_pred = clf.predict(X_list_real_test_raw)

logger.info("end predict for X_list_real_test_raw")
"""


print("\n\n\n\n")

#print("Y_test = \n\n", Y_test, type(Y_test))

print("\n\n\n\n")



list_Y_test_pred = Y_test_pred.tolist()

#print("list_Y_test_pred = \n\n", list_Y_test_pred, type(list_Y_test_pred))

print("\n\n\n\n")


#Y_test_cv_pred

#list_Y_test_cv_pred = Y_test_cv_pred.tolist()

#print("list_Y_test_cv_pred = \n\n", list_Y_test_cv_pred, type(list_Y_test_cv_pred))




print("\n\n\n\n")

list_Y_real_test_pred = Y_real_test_pred.tolist()


#print("list_Y_real_test_pred = \n\n", list_Y_real_test_pred, type(list_Y_real_test_pred))


print("\n\n\n\n")


print("len of list_Y_real_test_pred: ", len(list_Y_real_test_pred))



print("\n\n\n\n")





cur_time = int(time.time())

print("cur_time = ", cur_time)


#


str_fn_postfix = str_arg_1 + "_" + str_arg_2 + "_" + str(i_arg_4) + "gram"


str_file_name_tw = "../csv/group36_" + "logistic_regression" + "_" + str(cur_time) + "_submission_" + str_fn_postfix + ".csv"

str_file_name_prob_tw = "../csv/group36_" + "logistic_regression" + "_" + str(cur_time) + "_submission_" + str_fn_postfix + "_prob" + ".csv"

csv_stat_all_data = []

csv_stat_all_data_prob = []

i_index = 0


for item_predict in Y_list_raw_real_test:
    
    csv_stat_item = {}
    
    #tmp_pure_file_name = all_pure_files_list_real_test_raw[i_index]
    #wlist = tmp_pure_file_name.split('.')
    
    csv_stat_item['id'] = i_index

    csv_stat_item['category'] = item_predict
    #csv_stat_item['category'] = item_predict
    
    """
    csv_stat_item_prob = {}
    
    csv_stat_item_prob['id'] = i_index
    csv_stat_item_prob['category'] = item_predict
    
    list_probs = list_Y_real_test_pred_prob[i_index]
    
    for idx in range(0, i_lable_cnt, 1):
        str_title = "l_" + str(idx + 1)
        csv_stat_item_prob[str_title] = list_probs[idx]
    """
    
    i_index += 1
     
    csv_stat_all_data.append(csv_stat_item)
    
    #csv_stat_all_data_prob.append(csv_stat_item_prob)


i_index = 0

for item_predict in list_Y_real_test_pred:
    
    """
    csv_stat_item = {}
    
    #tmp_pure_file_name = all_pure_files_list_real_test_raw[i_index]
    #wlist = tmp_pure_file_name.split('.')
    
    csv_stat_item['id'] = i_index

    csv_stat_item['category'] = item_predict
    #csv_stat_item['category'] = item_predict
    """
    
    
    csv_stat_item_prob = {}
    
    csv_stat_item_prob['id'] = i_index
    csv_stat_item_prob['category'] = item_predict
    
    list_probs = list_Y_real_test_pred_prob[i_index]
    
    for idx in range(0, i_lable_cnt, 1):
        str_title = "l_" + str(idx + 1)
        csv_stat_item_prob[str_title] = list_probs[idx]
    
    
    i_index += 1
     
    #csv_stat_all_data.append(csv_stat_item)
    
    csv_stat_all_data_prob.append(csv_stat_item_prob)



csv_set_sorted = sorted(csv_stat_all_data, key=lambda x: (x['id']), reverse=False)



with open(str_file_name_tw, 'wt', newline='') as csvfile:
    
    csv_writer = csv.writer(csvfile, dialect='excel')
    
    
    header = ['Id', 'Category']
    
    
    
    csv_writer.writerow(header)
    
    for data_point in csv_set_sorted:
        
        row = [data_point['id'], data_point['category']]
        #str_to_write = "\n" + str(data_point['id']) + "," + str(data_point['category'])
        #fout_tw.write(str_to_write)
        csv_writer.writerow(row)
       
csv_set_sorted_prob = sorted(csv_stat_all_data_prob, key=lambda x: (x['id']), reverse=False)


with open(str_file_name_prob_tw, 'wt', newline='') as csvfile:
    
    csv_writer = csv.writer(csvfile, dialect='excel')
    
    
    header = ['Id', 'Category']
    
    for idx in range(0, i_lable_cnt, 1):
        header.append("l_" + str(idx + 1))
    
    csv_writer.writerow(header)
    
    
        
        
    for data_point in csv_set_sorted_prob:
        
        row = [data_point['id'], data_point['category']]
        for idx in range(0, i_lable_cnt, 1):
            str_title = "l_" + str(idx + 1)
            row.append(data_point[str_title])
        
        #str_to_write = "\n" + str(data_point['id']) + "," + str(data_point['category'])
        #fout_tw.write(str_to_write)
        csv_writer.writerow(row)
       

logger.info("program ends. ")
