from conllu import *
import json
from general_utils import *

def convert_request(req):

    lines=[]
    lines.append(f"# text = {req['sentence']}")
    curr_p=1
    words=req['wordPositions']
    root_pos=0
    w1=[]
    w2=[]

    empty_template='\t'.join(['_']*9)

    for word in words:
        word=word[0] #is list necessary
        if "link" in word.keys():
            feats=word['FEAT'][1:] #not vlear
            if len(feats)==0:
                feats='_'
            else:
                feats=feats[0]
            w=[curr_p,word['word'],'_',word['FEAT'][0],'_',feats,0,word['link'],'_','_']
            if 'dep_word' in word.keys():
                w1=w
            if 'main_word' in word.keys():
                root_pos=curr_p
                w[-3]='_'
                w2=w
                
        else:
            header=f"{curr_p}\t"
            header+=empty_template
            lines.append(header)
        curr_p+=1

    w1[6]=root_pos

    w1=[str(x) for x in w1]
    w2=[str(x) for x in w2]

    lines.append('\t'.join(w1))
    lines.append('\t'.join(w2))
    return lines

def to_conllu(request,all_features=None,all_rel=None,all_pos=None,all_words=None):
    with open('req.json',encoding='utf-8') as f:
        request=json.load(f)
    
    lines=convert_request(request)

    d=ConlluDataset()
    d.compute_from_lines(lines)

    '''
    self.all_features={}
    self.all_words=[]
    self.all_pos=[]
    self.all_rel=[]

    '''
    if all_features:
        d.all_features=all_features

    if all_rel:
        d.all_rel=all_rel

    if all_pos:
        d.all_pos=all_pos

    if all_words:
        d.all_words=all_words #if bag of words

    #all_rel - load from bow (sum)
    #all_features - bow_0
    #all_pos - same


    return d

#NOT TESTEF

from syntagrusbin import *

def bin_conllu(con_req,bin_example=None,masks=None):
    if not bin_example:
        raise ValueError("Initiate binarizer first")
    if not masks:
        raise ValueError("pass rel mask as parameter")

    bin_example.load_data(con_req)
    bin_example.all_rel=mask
    bin_example.data.all_rel=mask
    bin_example._bag_of_words_rel_cache_init()
    bin_example.generate_dataset(ratio=0)

    return bin_example.sources

from models import *
import joblib

def evaluate(req_bin,model_dir,model_idx,target_idx):
    model=joblib.load(model_dir+f'/{model_idx}_model.pth')
    scaler=joblib.load(model_dir+f'/{model_idx}_skaler.pkl')
    model.scaler=scaler
    result=model.predict(req_bin)
    #target_idx - idx of given rel, relative, not global
    return result, result[target_idx]

if __name__=="__main__":

    res=to_conllu('req.json')
    print(res.tokens)
