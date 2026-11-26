from conllu import *
import json
from general_utils import *

from syntagrusbin import *
    
from models import *
import joblib
import os

class ReqProcessor:
    def __init__(self,directory):
        self.directory=directory

        config=f'{self.directory}/0.json'
        with open(config,encoding='utf-8') as f:
            config=json.load(f)

        self.bin_=SynTagRusBinarizer(use_bert=config["use_bert"],bert_name=config["bert_name"], batch_size=2,
                                    unk_token= config["unk_token"], use_head= config["use_head"], predict_head = False,
                                    use_pos= config["use_pos"], use_morph= config["use_morph"], use_sentenceemb= config["use_sentenceemb"],
                                    use_s_no= config["use_s_no"], use_distance= config["use_distance"], binary= config["binary"], sentence_batch=1
                                    )

        self.w1=''
        self.w2=''
        self.rel=''

    def convert_request(self,req):

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
                    self.w1=word['word']
                    self.rel=word['link']
                if 'main_word' in word.keys():
                    root_pos=curr_p
                    w[-3]='_'
                    w2=w
                    self.w2=word['word']
                    
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

    def to_conllu(self,request,all_features=None,all_rel=None,all_pos=None,all_words=None):
        
        lines=self.convert_request(request)

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


    def bin_conllu(self,con_req,masks=None):
        if not masks:
            raise ValueError("pass rel mask as parameter")

        self.bin_.load_data(con_req)
        self.bin_.all_rel=masks
        self.bin_.data.all_rel=masks
        self.bin_._bag_of_words_rel_cache_init()
        self.bin_.generate_dataset(ratio=0)
        #print(self.bin_.data.tokens)

        return self.bin_.sources


    def evaluate(self,req_bin,model_idx):
        model=joblib.load(self.directory+f'/{model_idx}_model.pth')
        scaler=joblib.load(self.directory+f'/{model_idx}_scaler.pkl')
        model.scaler=scaler
        result=model.predict_proba(req_bin)
        #target_idx - idx of given rel, relative, not global
        return result

    def one_request(self,request):

        target_rel=request['wordPositions']
        target_rel=[x[0]['link']for x in target_rel  if 'link' in x[0] ][0]

        n_pred,idx=find_idx(target_rel,self.directory)

        bow=f'{self.directory}/{n_pred}_bow.json'
        with open(bow,encoding='utf-8') as f:
            bow=json.load(f)

        config=f'{self.directory}/{n_pred}.json'
        with open(config,encoding='utf-8') as f:
            config=json.load(f)


        all_features=None
        all_rel=bow['all_rel']
        all_pos=None
        all_words=None

        if config['use_pos']:
            all_pos=bow['all_pos']
        if config['use_morph']:
            all_features=bow['all_features']
        if not config['use_bert']:
            all_words=bow['all_words']


        res=self.to_conllu(request,all_features,all_rel,all_pos,all_words)
        target_rel=[x['relation']  for x in res.tokens if x['relation']!='_'][0]
        

        src=self.bin_conllu(res,bow['all_rel'])
        #print(src)

        vector=self.evaluate(src,n_pred)[0]
        return vector[idx],self.w1,self.w2,self.rel



if __name__=="__main__":
    '''with open('req.json',encoding='utf-8') as f:
        request=json.load(f)'''

    directory=input('model directory: ')
    pipe_p=input('pipe path: ')

    pr=ReqProcessor(directory)
    '''reqs=[request]
    for i in reqs:
        r=pr.one_request(i)
        print(r)'''

    if not os.path.exists(pipe_p):
        os.mkfifo(pipe_p)

    with open(pipe_p,"r") as pipe:
        request=pipe.read()
        request=json.loads(request)
        res=pr.one_request(request)
        print(f"The probability of words {res[1]} and {res[2]}'s dependency to be {res[3]} is {res[0]:5f}")

    os.unlink(pipe_p)
    pr.bin_._unload_bert()