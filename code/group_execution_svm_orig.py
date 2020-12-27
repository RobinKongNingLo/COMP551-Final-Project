# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:45:00 2019

@author: PeterXu-Desktop
"""




import sys
#import subprocess

#subprocess.call(["./sample_to_execute.py", "a", "b"])



orig_stdout = sys.stdout

i_max_feature_size = 0
i_gram = 0

for str_data_set in ("imdb",):
#for str_data_set in ("imdb", "yelp"):
#for str_data_set in ("yelp", "imdb"):
    for b_tfidf_or_bow in (True, False):

        for i_gram in range(2, 0, -1):
        #for i_gram in range(5, 0, -1):
            
            if i_gram == 1:
                i_max_feature_size = 50000
            elif i_gram > 1:
                i_max_feature_size = 500000
            else:
                pass
            
            str_i_or_b = ""
            
            #rst_svm_imdb_1gram_bow.txt
            
            str_output_fn = "rst_svm_" + str_data_set + "_" + str(i_gram) + "gram_"
            if b_tfidf_or_bow:
                str_output_fn += "tfidf"
                str_i_or_b = "tfidf" 
            else:
                str_output_fn += "bow"
                str_i_or_b = "bow" 
                
            str_output_fn += "_orig.txt"
            
                       
            f = open(str_output_fn, 'w')
            
            sys.stdout = f
            
            ##sys.argv = ['./text_classfication_logistic_regression_v7.py', 'imdb_or_yelp', 'tfidf_or_bow', 'i_max_feature_size', 'i_gram']

            sys.argv = ['./text_classfication_svm_v8.py', str_data_set, str_i_or_b, str(i_max_feature_size), str(i_gram)]

            exec(open("./text_classfication_svm_v8.py").read())

            #print(i_gram, i_max_feature_size, str_data_set, b_tfidf_or_bow, str_output_fn, "\n")
            f.close()

"""

f = open('out.txt', 'w')

sys.stdout = f

##sys.argv = ['./text_classfication_logistic_regression_v7.py', 'imdb_or_yelp', 'tfidf_or_bow', 'i_max_feature_size', 'i_gram']


#sys.argv = ['./text_classfication_logistic_regression_v7.py', 'imdb', 'tfidf', '50000', '1']

exec(open("./text_classfication_logistic_regression_v7.py").read())

"""

















sys.stdout = orig_stdout

