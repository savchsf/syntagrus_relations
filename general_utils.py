from glob import glob
import os
import json

def get_prefixes(directory):
    paths=glob(directory+'/*.json') #counting prefixes by json
    prefixes=[os.path.split(x)[-1].replace('.json','') for x in list(paths) if '_bow' not in x]
    prefixes=[int(x) for x in prefixes]
    return prefixes

def to_absolute(idx,pred,directory):

    if int(pred)==0:
        return idx

    if idx==0:
        return idx

    prefs=get_prefixes(directory)
    add=0
    n=len(prefs)
    scheme={}

    for i in range(1,n):
        with open(directory+f'/{i-1}.json',encoding='utf-8') as f:
            scheme=json.load(f)

        curr_len=len(scheme['outputs'])-1

        add+=curr_len

        if int(pred)==i:
            return idx+add

    return 0


def find_idx(rel,directory):
    #return: (n_pred,idx)
    n_pred=0
    idx=0
    prefs=get_prefixes(directory)
    scheme={}
    
    for i in prefs:
        with open(directory+f'/{i}.json',encoding='utf-8') as f:
            scheme=json.load(f)
        rels=scheme['outputs']
        for j in rels:
            if rel==rels[j]:
                #print(rels)
                idx=int(j)
                n_pred=i
                break
    
    return n_pred,idx


if __name__=='__main__':
    print(find_idx('nmod','testt/binarized_data_val'))
    print(to_absolute(1,4,'testt/binarized_data_val'))