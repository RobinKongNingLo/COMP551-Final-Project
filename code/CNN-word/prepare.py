from utils import *

import os

from collections import Counter


import json


def load_data():
    data = []
    cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX} # char_to_idx
    wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX} # word_to_idx
    tti = {} # tag_to_idx
    
    s_train_data_fn = sys.argv[1]
    print("s_train_data_fn = ", s_train_data_fn, "\n")


    
        
    b_do_kaggle = False
    #b_do_kaggle = True
    
    #b_do_colab = False
    
    
    
    
    i_cross_validation_fold_count = 5
    
    #b_partial = True
    #b_partial = False
    
    
    
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
    
    
    
    #i_partial_count = 10000
    
    
    b_do_model_selection = False
    #b_do_model_selection = True
    
    b_do_cross_validation = False
    #b_do_cross_validation = True
    
    
    #b_use_new_data_set = False
    #b_use_new_data_set = True
    
    #b_use_new_data_IMDB_or_YELP = True
    #b_use_new_data_IMDB_or_YELP = False
    
    
    
    print("os.getcwd():", os.getcwd(), "\n")
    
    s_root = ""
    
    
    
    s_log_config_fn = "logging.conf"
    
    
    if b_do_colab:
    
        s_root = "/content/drive/My Drive/gcolab/comp511prj4/"
        s_log_config_fn = s_root + s_log_config_fn
    
    
    
    if b_do_kaggle:
        s_log_config_fn = "../input/config/" + s_log_config_fn
    
    #logging.config.fileConfig(s_log_config_fn)
    
    
    
    
    
    # create logger
    #logger = logging.getLogger('Project4Group36')
    
    
    
    #logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")
    
    
    str_fn_postfix = ""
    
    
    #b_Tfidf_or_BOW = False
    
    b_Tfidf_or_BOW = True
    
    i_ngram_range_upbound = 4
    
    i_max_features_cnt = None
    
    
    print("b_Tfidf_or_BOW = ", b_Tfidf_or_BOW, "\n")
    
    print("i_ngram_range_upbound = ", i_ngram_range_upbound, "\n")
    
    
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
    
    if b_do_colab:
        
        if b_use_new_data_set:
            #pass
            str_input_folder = "/content/drive/My Drive/gcolab/dataset/datasets/"
        else:
            str_json_fn_training = "/content/drive/My Drive/gcolab/dataset/imdb_2_labels/imdb_2_labels/" + "training" + "_set" + str_fn_postfix + ".json"
    
    
    
    if b_use_new_data_set:
        
        if b_use_new_data_IMDB_or_YELP:
            str_json_fn_training = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
        else:   
            str_json_fn_training = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"
    else:
        pass
        
    
    
    str_json_fn_testing = "../" + "testing" + "_set" + str_fn_postfix + ".json"
    
    if b_use_new_data_set:
        
        if b_use_new_data_IMDB_or_YELP:
            str_json_fn_testing = str_input_folder + "datasets/JMARS_10_label_imdb_dataset/data/data.json"
        else:   
            str_json_fn_testing = str_input_folder + "datasets/Zhang_5_label_yelp_dataset/data/data.json"
    
    
    print("str_json_fn_training = ", str_json_fn_training)
    
    #logger.info("str_json_fn_testing = %s \n", str_json_fn_testing)

        
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
            
        #val_len = i_partial_count * 0.2
        
        
    i_split_pos = int(float(len(real_training_data_set)) * 0.8)
    
    print("\n i_split_pos = ", i_split_pos)
    
    tmp_train_set = real_training_data_set[0:i_split_pos]
    tmp_vali_set = real_training_data_set[i_split_pos:]
    
    print("\n", len(tmp_train_set), len(tmp_vali_set))
    
    real_training_data_set = tmp_train_set[:]
    
    s_postfix = "_"
    if b_use_new_data_set:
        s_postfix += "t"
    else:
        s_postfix += "f"
        
    if b_use_new_data_IMDB_or_YELP:
        s_postfix += "t"
    else:
        s_postfix += "f"
        
    
    s_folder = ""
    if b_do_colab:
        s_folder = "/content/drive/My Drive/gcolab/comp511prj4/cnn_code/"
    
    with open(s_folder + "validation_data" + s_postfix + ".json", "w") as wf_json_set:
        json.dump(tmp_vali_set, wf_json_set)

    #fo = open(s_train_data_fn)
    
    
    reviews = []
    for data_point in real_training_data_set:
    
        x_orig = ""
        
        
        if b_use_new_data_set:
            x_orig = data_point['review']
        else:
            x_orig = data_point['simplified_cleanup']
        
        
        x_orig = str(x_orig)
        
                
        x = tokenize(x_orig, UNIT)
        reviews.append(" ".join(x))
        
    
    
    print ('Number of reviews :', len(reviews))
    print(reviews[1])
    #print("reviews = ", reviews)


    #all_words = ' '.join("%s" %id for id in reviews)
    all_words = ' '.join(reviews)
    #all_words = ' '.join(reviews)
    # create a list of words
    word_list = all_words.split()
    
    #print("\n word_list = ", word_list)
    
    # Count all the words using Counter Method
    count_words = Counter(word_list)
    
    #print (count_words[0:3])
    
    len_words = len(word_list)
    
    print("len_words = ", len_words)
    
    
    len_counted_words = len(count_words)
    
    print("len_counted_words = ", len_counted_words)
    
        
    #i_len_picked_words_len = 5000
    #i_len_picked_words_len = len_counted_words
    
    sorted_words = count_words.most_common(i_len_picked_words_len)
    
    print("type sorted_words = ", type(sorted_words))
    
    len_sorted_words = len(sorted_words)
    
    print("len_sorted_words = ", len_sorted_words)
    
    
    sorted_word_list = [w for i, (w,c) in enumerate(sorted_words)]
    
    #print("\n sorted_word_list = ", sorted_word_list)
    
    set_sorted_word = set(sorted_word_list)
    
    i_idx = 0
    for data_point in real_training_data_set:
    
    #for line in fo:
        #x, y = line.split("\t")
        
        x_orig = ""
        
        
        if b_use_new_data_set:
            x_orig = data_point['review']
        else:
            x_orig = data_point['simplified_cleanup']
        
        
        x_orig = str(x_orig)
        
        y_orig = ""
        
        
        if b_use_new_data_set:
            y_orig = data_point['rating']
            
        else:
            y_orig = data_point['y_val']
        
        
        if i_idx == 5:
            print("\n x_orig, y_orig before tokenize:", x_orig, y_orig)
           
        """
        if i_idx <= 519751 and i_idx >= 519741:
            print("\n i_idx, x, y before tokenize:", i_idx, x, y)
        """
        
        x = tokenize(x_orig, UNIT)
        
        
        
        
        y = str(y_orig)
        
        
        
        if i_idx == 5:
            print("\n x, y after tokenize:", x, y)
        
        """
        if i_idx <= 519751 and i_idx >= 519741:
            print("\n i_idx, x, y after tokenize:", i_idx, x, y)
        """ 
            
        
        
        y = y.strip()
        
        
        b_skip = False
        
        for w in x:
            
            if w == '':
                b_skip = True
                print("\n encounter special case: i_idx = ", i_idx)
                print("\n i_idx, w, x_orig, y_orig before tokenize:", i_idx, w, x_orig, y_orig)
                print("\n i_idx, w, x, y after tokenize:", i_idx, w, x, y)
                
                
                break
            
            #print("sorted_words = ", sorted_words)
            
            
            if w not in set_sorted_word:
                continue
            else:
                #print("w in sorted_words", w, sorted_words)
                pass
            
            for c in w:
                if c not in cti:
                    cti[c] = len(cti)
            if w not in wti:
                wti[w] = len(wti)
                
                """
                if wti[w] <= 20117 and wti[w] >= 20113:
                   
                    print("\n i_idx, w, wti w, x_orig, y_orig before tokenize:", i_idx, w, wti[w], x_orig, y_orig)
                    print("\n i_idx, w, wti w, x, y after tokenize:", i_idx, w, wti[w], x, y)
                """
                    
        
        if b_skip:
            continue
        
        
                
        if y not in tti:
            tti[y] = len(tti)

    
        #x = ["+".join(str(cti[c]) for c in w) + ":%d" % wti[w] for w in x]
        
        x_l = []
        for w in x:
            if w in set_sorted_word:
                w_item = "+".join(str(cti[c]) for c in w) + ":" + str(wti[w])
                x_l.append(w_item)
            else:
                pass

        if len(x_l) == 0:
            continue
        
        x = x_l[:]
        
        y = [str(tti[y])]
        
        if i_idx == 10:
            print("\n x after , y after ", x, y)
        
        data.append(x + y)
        
        i_idx += 1
        
        if i_idx % 10000 == 0:
            logger.info("i_idx = %d", i_idx)
        
    #fo.close()
    data.sort(key = len, reverse = True)
    return data, cti, wti, tti

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: %s training_data" % sys.argv[0])
    
    
    logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")
    
    
    data, cti, wti, tti = load_data()
    
    s_folder = ""
    if b_do_colab:
        s_folder = "/content/drive/My Drive/gcolab/comp511prj4/cnn_code/"
       
    s_postfix = "_"
    if b_use_new_data_set:
        s_postfix += "t"
    else:
        s_postfix += "f"
        
    if b_use_new_data_IMDB_or_YELP:
        s_postfix += "t"
    else:
        s_postfix += "f"
        
    
    save_data(s_folder + sys.argv[1] + s_postfix + ".csv", data)
    save_tkn_to_idx(s_folder + sys.argv[1] + s_postfix + ".char_to_idx", cti)
    save_tkn_to_idx(s_folder + sys.argv[1] + s_postfix + ".word_to_idx", wti)
    save_tkn_to_idx(s_folder + sys.argv[1] + s_postfix + ".tag_to_idx", tti)

    
    logger.info("\n\n\n\n\n\n\n\n\n\nprogram ends. ")
    