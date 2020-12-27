from model import *
from utils import *
from evaluate import *

def load_data():
    data = []
    bxc = [] # character sequence batch
    bxw = [] # word sequence batch
    by = [] # label batch
    
    #b_do_colab = False
    #b_do_colab = True
    
    str_input_folder = "./"
    if b_do_colab:
        str_input_folder = "/content/drive/My Drive/gcolab/comp511prj4/cnn_code/"


    s_postfix = "_"
    if b_use_new_data_set:
        s_postfix += "t"
    else:
        s_postfix += "f"
        
    if b_use_new_data_IMDB_or_YELP:
        s_postfix += "t"
    else:
        s_postfix += "f"
      
        
    cti = load_tkn_to_idx(str_input_folder + "training_data" + s_postfix + "." + sys.argv[2]) # char_to_idx
    wti = load_tkn_to_idx(str_input_folder + "training_data" + s_postfix + "." + sys.argv[3]) # word_to_idx
    itt = load_idx_to_tkn(str_input_folder + "training_data" + s_postfix + "." + sys.argv[4]) # idx_to_tkn
    print("loading %s" % sys.argv[5] + s_postfix + ".csv")
    
    
    logger.info("\n before loading %s. ", sys.argv[5])
    
    i_idx = 0
    
    fo = open(str_input_folder + sys.argv[5] + s_postfix + ".csv", "r")
    for line in fo:
        
        #print("\n i_idx = ", i_idx)
        
        line = line.strip()
        *x, y = [x.split(":") for x in line.split(" ")]
        
        
        #xc_coll, xw_coll = zip(*[(list(map(int, xc.split("+"))), int(xw)) for xc, xw in x])
        
        
        xc_coll = []
        xw_coll = []
        
        for xc, xw in x:
                        
            converted_xw = 0          
        
            try:
                #int('')
                converted_xw = int(xw)
            except ValueError:
                #pass      # or whatever
                print("\n exception occur i_idx = ", i_idx)
                
                
                print("\n line: ", line)
                print("\n x, y: ", x, y)
                print("\n xc_coll: ", xc_coll)
                print("\n xw_coll: ", xw_coll)
                
            converted_xc = None
            
            
            
            
            try:
                #int('')
                converted_xc = list(map(int, xc.split("+")))
            except ValueError:
                #pass      # or whatever
                print("\n exception occur i_idx = ", i_idx)                
                
                print("\n line: ", line)
                print("\n x, y: ", x, y)
                print("\n xc_coll: ", xc_coll)
                print("\n xw_coll: ", xw_coll)
            
            
            xc_coll.append(converted_xc)
            
            xw_coll.append(converted_xw)
        
        
        xc_coll = tuple(xc_coll)
        xw_coll = tuple(xw_coll)      
        
        
        if i_idx == 10:
            print("\n line: ", line)
            print("\n x, y: ", x, y)
            print("\n xc_coll: ", xc_coll)
            print("\n xw_coll: ", xw_coll)
        
        
        bxc.append(xc_coll)
        bxw.append(xw_coll)
        by.append(int(y[0]))
        if len(by) == BATCH_SIZE:
            bxc, bxw = batchify(bxc, bxw, max(KERNEL_SIZES))
            data.append((bxc, bxw, LongTensor(by)))
            bxc = []
            bxw = []
            by = []
            
        i_idx += 1
        
    fo.close()
    
    
    
    logger.info("\n end loading %s. ", sys.argv[5])
    
    
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, cti, wti, itt

def train():
    
    s_postfix = "_"
    if b_use_new_data_set:
        s_postfix += "t"
    else:
        s_postfix += "f"
        
    if b_use_new_data_IMDB_or_YELP:
        s_postfix += "t"
    else:
        s_postfix += "f"
    
    
    num_epochs = int(sys.argv[-1])
    data, cti, wti, itt = load_data()
    model = cnn(len(cti), len(wti), len(itt))
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    epoch = load_checkpoint(sys.argv[1], model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", sys.argv[1])
    print("training model...")
    for ei in range(epoch + 1, epoch + num_epochs + 1):
        
        logger.info("ei = %d. ", ei)
        
        loss_sum = 0
        timer = time.time()
        for xc, xw, y in data:
            model.zero_grad()
            loss = F.nll_loss(model(xc, xw), y) # forward pass and compute loss
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = loss.tolist()
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(data)
        if ei % SAVE_EVERY and ei != epoch + num_epochs:
            save_checkpoint("", None, ei, loss_sum, timer)
        else:
            save_checkpoint(filename, model, ei, loss_sum, timer)
    


        if EVAL_EVERY and (ei % EVAL_EVERY == 0 or ei == epoch + num_epochs):
            args = [model, cti, wti, itt]
            
            
    
            logger.info(" before evaluate %s. ", sys.argv[6])
    

            evaluate(predict(sys.argv[6] + s_postfix + ".json", *args), True)
            
            
            
            model.train()
            print()

if __name__ == "__main__":
    if len(sys.argv) not in [7, 8]:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx training_data (validation_data) num_epoch" % sys.argv[0])
        
        
        """
        sys.argv[2] += s_postfix
        sys.argv[3] += s_postfix
        sys.argv[4] += s_postfix
        sys.argv[5] += s_postfix + ".csv"
        sys.argv[6] += s_postfix + ".json"
        """
        
    print("cuda: %s" % CUDA)
    train()
