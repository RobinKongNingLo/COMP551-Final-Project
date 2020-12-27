from predict import *

from sklearn.metrics import f1_score

from sklearn import metrics

def evaluate(result, summary = False):
    
    
    y_true = []
    y_pred = []
    
    avg = defaultdict(float) # average
    tp = defaultdict(int) # true positives
    tpfn = defaultdict(int) # true positives + false negatives
    tpfp = defaultdict(int) # true positives + false positives
    for _, y0, y1, _ in result: # actual value, prediction
        
        y_true.append(y0)
        y_pred.append(y1)
        
        tp[y0] += y0 == y1
        tpfn[y0] += 1
        tpfp[y1] += 1
        
    for y in sorted(tpfn.keys()):
        pr = tp[y] / tpfp[y] if tpfp[y] else 0
        rc = tp[y] / tpfn[y] if tpfn[y] else 0
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        if not summary:
            print()
            print("label = %s" % y)
            print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
            print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
            print("f1 = %f" % f1(pr, rc))
    avg["macro_pr"] /= len(tpfn)
    avg["macro_rc"] /= len(tpfn)
    avg["micro_f1"] = sum(tp.values()) / sum(tpfp.values())
    print()
    #print("macro precision = %f" % avg["macro_pr"])
    #print("macro recall = %f" % avg["macro_rc"])
    #print("macro f1 = %f" % f1(avg["macro_pr"], avg["macro_rc"]))
    #print("micro f1 = %f" % avg["micro_f1"])
    
    
    
    #print('macro precision: %f' % avg["macro_pr"], '| macro recall: %f' % avg["macro_rc"], '| macro f1: %f' % f1(avg["macro_pr"], avg["macro_rc"]), '| micro f1: %f ' % avg["micro_f1"])
    #print('macro precision: %.8f' % avg["macro_pr"])
    
    

    #print("\n result:", type(result))
    
    
    i_index = 0
    #for _, y0, y1, _ in result: # actual value, prediction
        
        
        
        #i_index += 1
    
    #print("\n result:", y_true[0:5], y_pred[0:5])
    
    print("\n\n")
    print(metrics.classification_report(y_true, y_pred, digits=5), "\n")
    
    
    f1_macro = f1_score(y_true, y_pred, average='macro')  

    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    f1_weighted = f1_score(y_true, y_pred, average='weighted') 
    
    
    print('macro f1: %f' % f1_macro, '| micro f1: %f ' % f1_micro, '| weighted f1: %f ' % f1_weighted)
    
    
    

if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit("Usage: %s model char_to_idx word_to_idx tag_to_idx test_data" % sys.argv[0])
    print("cuda: %s" % CUDA)
    evaluate(predict(sys.argv[5], *load_model()))
