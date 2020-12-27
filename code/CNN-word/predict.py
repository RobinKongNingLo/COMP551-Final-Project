from model import *
from utils import *

import json

def load_model():
    cti = load_tkn_to_idx(sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(sys.argv[4]) # idx_to_tag
    model = cnn(len(cti), len(wti), len(itt))
    print(model)
    load_checkpoint(sys.argv[1], model)
    return model, cti, wti, itt

def run_model(model, itt, batch):
    batch_size = len(batch) # real batch size
    while len(batch) < BATCH_SIZE:
        batch.append([-1, "", [], [], ""])
    batch.sort(key = lambda x: -len(x[2]))
    xc, xw = batchify(*zip(*[(x[2], x[3]) for x in batch]), max(KERNEL_SIZES))
    result = model(xc, xw)
    for i in range(batch_size):
        y = itt[result[i].argmax()]
        p = result[i].max().exp().item()
        batch[i].append(y)
        batch[i].append(p)
        if VERBOSE:
            print()
            print(batch[i])
            y = torch.exp(result[i]).tolist()
            for j, p in sorted(enumerate(y), key = lambda x: -x[1]):
                print("%s %f" % (itt[j], p))
    return [(x[1], *x[4:]) for x in sorted(batch[:batch_size])]

def predict(filename, model, cti, wti, itt):
    data = []
    
    
    #b_use_new_data_set = False
    
    #fo = open(filename)
    str_json_fn_vali = filename
    
    #b_do_colab = False
    #b_do_colab = True
    
    str_input_folder = "./"
    if b_do_colab:
        str_input_folder = "/content/drive/My Drive/gcolab/comp511prj4/cnn_code/"
    
    str_json_fn_vali = str_input_folder + str_json_fn_vali
    
    #print("str_json_fn_vali = ", str_json_fn_vali)
    
    
    
    #logger.info("str_json_fn_testing = %s \n", str_json_fn_testing)

    logger.info(" before loading %s. ", str_json_fn_vali)
        
    jsonFile = open(str_json_fn_vali, "r") # Open the JSON file for reading
    json_data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file
    
    i_index = 0
    
    
    for data_point in json_data:
    #for idx, line in enumerate(fo):
           
        #line = line.strip()
        
        if b_use_new_data_set:
            x = data_point['review']
        else:
            x = data_point['simplified_cleanup']
        
        
        x = str(x)
        
        y = ""
        
        
        if b_use_new_data_set:
            y = data_point['rating']
            
        else:
            y = data_point['y_val']
        
        y = str(y)
        
        
        #line, y = line.split("\t") if line.count("\t") else [line, None]
        
        
        x = tokenize(x, UNIT)
        xc = [[cti[c] if c in cti else UNK_IDX for c in w] for w in x]
        xw = [wti[w] if w in wti else UNK_IDX for w in x]
        data.append([i_index, x, xc, xw, y])
        
        i_index += 1
    #fo.close()
    
    logger.info(" end loading %s. ", str_json_fn_vali)
    
    with torch.no_grad():
        model.eval()
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            for y in run_model(model, itt, batch):
                yield y

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    result = predict(sys.argv[5], *load_model())
    print()
    for x, y0, y1, p in result:
        print((x, y0, y1, p) if y0 else (x, y1, p))
